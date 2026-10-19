"""Interactive Casimir explorer: a pop-up window with tunable parameters.

    python visualize.py          (or:  python main.py --gui)

Left    the lattice between the two plates. Links are arrows coloured by the
        electric field <S^z>, plaquettes are tiles coloured by the magnetic
        energy density J <U_p + U_p^dag>, plate sites show their conserved
        static charges. The rightmost column of hollow sites is the periodic
        image of the first.
Right   (a) energy density across the gap against the bulk reference,
        (b) model-free Casimir signal D2(d) = E0(d+1) - 2 E0(d) + E0(d-1) and the
            in-sector gap for every separation at the current couplings,
        (c) real-time traces when the quench is running.
Bottom  sliders for W, d, J, g, lam_RK; radio buttons for the field pattern, the
        plate model and the view; buttons to run/pause a quench and save a PNG.

"Quench" starts from the most-flippable product state (the initial state of
the ground-state filter and of the hardware quench) and evolves it exactly
under the current Hamiltonian, so the build-up of vacuum fluctuations between
the plates can be watched directly. The same quench is studied quantitatively
in dynamics.py and run as a circuit by hardware.py.

All numerics run in the gauge-invariant sector (sector.py), so the window stays
responsive up to roughly 40 link qubits. The class works with any matplotlib
backend; the test-suite drives it headlessly with Agg.
"""

from __future__ import annotations

import numpy as np

from lattice import LatticeGrid
from QuantumFields.gauge_field import default_plates
from sector import GaugeSector

MAX_DIM = 250_000          # largest sector the GUI will diagonalise
CURVE_DIM = 60_000         # largest sector used for the E0(d) curve
DENSE_DYNAMICS = 2500      # exact eigendecomposition below this dimension


class CasimirModel:
    """Numerics behind the window, cached per parameter set."""

    def __init__(self):
        self._sectors = {}
        self._ground = {}

    @staticmethod
    def resolve_plates(width, plates):
        """"auto" picks the default plate model; fixed staggered charges need even W."""
        if plates == "auto" or (plates == "fixed" and width % 2):
            return default_plates(width)
        return plates

    def sector(self, W, H, plates):
        key = (W, H, plates)
        if key not in self._sectors:
            self._sectors[key] = GaugeSector(LatticeGrid(W, H), plates=plates,
                                             max_dim=MAX_DIM)
        return self._sectors[key]

    def ground(self, W, H, plates, g, J, rk, pattern):
        key = (W, H, plates, round(g, 6), round(J, 6), round(rk, 6), pattern)
        if key not in self._ground:
            sector = self.sector(W, H, plates)
            gs = sector.ground_state(g=g, J=J, rk=rk, field_pattern=pattern)
            self._ground[key] = (sector, gs)
        return self._ground[key]

    def curve(self, W, plates, g, J, rk, pattern, d_max=10):
        """E0(d) and gap for every separation whose sector is small enough."""
        ds, E0, gaps = [], [], []
        for d in range(1, d_max + 1):
            try:
                sector = self.sector(W, d + 1, plates)
            except MemoryError:
                break
            if sector.dim > CURVE_DIM:
                break
            _, gs = self.ground(W, d + 1, plates, g, J, rk, pattern)
            ds.append(d), E0.append(gs['energy']), gaps.append(gs['gap'])
        return np.array(ds), np.array(E0), np.array(gaps)


class CasimirExplorer:
    """The interactive window (sliders, radio buttons, quench animation)."""

    def __init__(self, fig=None, W=4, d=3, J=1.0, g=0.0, rk=0.0,
                 pattern="uniform", plates="auto"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider

        self.plt = plt
        self.model = CasimirModel()
        self.params = dict(W=W, d=d, J=J, g=g, rk=rk, pattern=pattern, plates=plates)
        self.view = "ground state"
        self.playing = False
        self.time = 0.0
        self.dt = 0.08
        self.trace = {'t': [], 'HB': [], 'echo': []}
        self._dyn = None
        self._colorbars = []

        self.fig = fig or plt.figure(figsize=(15, 9))
        try:
            self.fig.canvas.manager.set_window_title("Casimir explorer: U(1) quantum link model")
        except Exception:
            pass
        gs = self.fig.add_gridspec(3, 2, left=0.05, right=0.97, top=0.95, bottom=0.37,
                                   width_ratios=[1.35, 1.0], hspace=0.55, wspace=0.18)
        self.ax_lat = self.fig.add_subplot(gs[:, 0])
        self.ax_prof = self.fig.add_subplot(gs[0, 1])
        self.ax_cas = self.fig.add_subplot(gs[1, 1])
        self.ax_cas_gap = self.ax_cas.twinx()
        self.ax_dyn = self.fig.add_subplot(gs[2, 1])
        self.cax_e = self.fig.add_axes([0.06, 0.295, 0.22, 0.012])
        self.cax_b = self.fig.add_axes([0.33, 0.295, 0.22, 0.012])
        self.status = self.fig.text(0.585, 0.05, "", family="monospace", fontsize=8.5,
                                    va="bottom")

        def slider(rect, label, lo, hi, val, step=None):
            ax = self.fig.add_axes(rect)
            return Slider(ax, label, lo, hi, valinit=val, valstep=step)

        self.s_W = slider([0.09, 0.195, 0.16, 0.02], "width W", 2, 6, W, 1)
        self.s_d = slider([0.09, 0.160, 0.16, 0.02], "separation d", 1, 8, d, 1)
        self.s_J = slider([0.09, 0.125, 0.16, 0.02], "plaquette J", 0.0, 4.0, J)
        self.s_g = slider([0.09, 0.090, 0.16, 0.02], "field g", 0.0, 6.0, g)
        self.s_rk = slider([0.09, 0.055, 0.16, 0.02], r"$\lambda_{RK}$", -1.0, 1.5, rk)
        self.s_speed = slider([0.09, 0.020, 0.16, 0.02], "time step", 0.01, 0.3, self.dt)
        for s in (self.s_W, self.s_d, self.s_J, self.s_g, self.s_rk):
            s.on_changed(self._on_param)
        self.s_speed.on_changed(lambda v: setattr(self, 'dt', float(v)))

        def radio(rect, title, labels, active):
            ax = self.fig.add_axes(rect)
            ax.set_title(title, fontsize=9)
            return RadioButtons(ax, labels, active=active)

        self.r_pattern = radio([0.30, 0.095, 0.075, 0.10], "electric field",
                               ("uniform", "staggered"), 0 if pattern == "uniform" else 1)
        self.r_plates = radio([0.385, 0.095, 0.075, 0.10], "plates",
                              ("auto", "free", "fixed"), ("auto", "free", "fixed").index(plates))
        self.r_view = radio([0.47, 0.095, 0.085, 0.10], "view",
                            ("ground state", "quench"), 0)
        self.r_pattern.on_clicked(self._on_param)
        self.r_plates.on_clicked(self._on_param)
        self.r_view.on_clicked(self._on_view)

        self.b_play = Button(self.fig.add_axes([0.30, 0.03, 0.075, 0.04]), "Play / pause")
        self.b_reset = Button(self.fig.add_axes([0.385, 0.03, 0.075, 0.04]), "Restart")
        self.b_save = Button(self.fig.add_axes([0.47, 0.03, 0.085, 0.04]), "Save PNG")
        self.b_play.on_clicked(self._on_play)
        self.b_reset.on_clicked(lambda _: self.restart_quench())
        self.b_save.on_clicked(lambda _: self.save("casimir_explorer.png"))

        self.timer = self.fig.canvas.new_timer(interval=60)
        self.timer.add_callback(self._tick)
        self.refresh()

    # ------------------------------------------------------------------
    # parameter handling
    # ------------------------------------------------------------------

    def set_params(self, **kwargs):
        """Programmatic parameter change (used by the tests and for figures)."""
        self.params.update(kwargs)
        self.refresh()

    def _read_widgets(self):
        self.params.update(W=int(self.s_W.val), d=int(self.s_d.val), J=float(self.s_J.val),
                           g=float(self.s_g.val), rk=float(self.s_rk.val),
                           pattern=self.r_pattern.value_selected,
                           plates=self.r_plates.value_selected)

    def _on_param(self, _):
        self._read_widgets()
        self.refresh()

    def _on_view(self, label):
        self.view = label
        self.playing = False
        self.timer.stop()
        if label == "quench":
            self.restart_quench()
        else:
            self.refresh()

    def _on_play(self, _):
        if self.view != "quench":
            self.r_view.set_active(1)              # fires _on_view("quench")
        self.playing = not self.playing
        (self.timer.start if self.playing else self.timer.stop)()

    # ------------------------------------------------------------------
    # numerics
    # ------------------------------------------------------------------

    def _solve(self):
        p = self.params
        plates = self.model.resolve_plates(p['W'], p['plates'])
        H = p['d'] + 1
        sector, gs = self.model.ground(p['W'], H, plates, p['g'], p['J'], p['rk'],
                                       p['pattern'])
        return plates, sector, gs

    def restart_quench(self):
        """Product state with the most flippable plaquettes, evolved exactly."""
        self.time = 0.0
        self.trace = {'t': [], 'HB': [], 'echo': []}
        self._dyn_lines = None
        try:
            plates, sector, gs = self._solve()
        except MemoryError as exc:
            self._message(str(exc))
            return
        idx = gs['support']
        Hs = gs['hamiltonian'][idx][:, idx]
        k = sector.most_flippable(idx, Hs)
        psi0 = np.zeros(len(idx), dtype=complex)
        psi0[k] = 1.0
        self._gs_flip = float(sector.flippability(gs['state']).mean())
        dyn = {'idx': idx, 'psi0': psi0, 'psi': psi0.copy(), 'H': Hs}
        if len(idx) <= DENSE_DYNAMICS:
            w, v = np.linalg.eigh(Hs.toarray())
            dyn['eig'] = (w, v, v.conj().T @ psi0)
        self._dyn = dyn
        self.refresh()

    def step(self, n=1):
        """Advance the quench by n time steps."""
        from scipy.sparse.linalg import expm_multiply
        dyn = self._dyn
        if dyn is None:
            return
        self.time += n * self.dt
        if 'eig' in dyn:
            w, v, c0 = dyn['eig']
            dyn['psi'] = v @ (np.exp(-1j * w * self.time) * c0)
        else:
            dyn['psi'] = expm_multiply(-1j * n * self.dt * dyn['H'], dyn['psi'])

    def _tick(self):
        if self.playing and self.view == "quench":
            self.step()
            self.refresh(light=True)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def _message(self, text):
        self.ax_lat.clear()
        self.ax_lat.axis("off")
        self.ax_lat.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
                         transform=self.ax_lat.transAxes, fontsize=12)
        self.fig.canvas.draw_idle()

    def refresh(self, light=False):
        try:
            plates, sector, gs = self._solve()
        except MemoryError as exc:
            self._message(f"Sector too large for interactive use:\n{exc}\n"
                          "Reduce W or d.")
            return
        p = self.params
        if self.view == "quench" and self._dyn is not None \
                and len(self._dyn['idx']) == len(gs['support']):
            vec = np.zeros(sector.dim, dtype=complex)
            vec[self._dyn['idx']] = self._dyn['psi']
            title = f"quench from the maximally flippable product state, t = {self.time:.2f}"
        else:
            vec = gs['state']
            title = "ground state"
        efield = sector.electric_field(vec)
        plaq = sector.plaquette_expectations(vec)
        flip = sector.flippability(vec)
        charges = sector.plate_charge_table().T @ (np.abs(vec) ** 2)

        quench = self.view == "quench"
        # During a quench the energy is conserved, so the tiles show the
        # flippability, which does evolve; otherwise the magnetic energy density.
        tiles = flip if quench else p['J'] * plaq
        self._draw_lattice(sector.lattice, efield, tiles, charges, title, light)
        self._draw_profile(sector.lattice, plaq, flip, plates, light)
        if not light:
            self._draw_casimir(plates)
        if self.view == "quench":
            self.trace['t'].append(self.time)
            self.trace['HB'].append(float(flip.mean()))
            self.trace['echo'].append(abs(np.vdot(self._dyn['psi0'], self._dyn['psi'])) ** 2
                                      if self._dyn is not None else 1.0)
        self._draw_dynamics(gs, light)
        if not light:
            self._draw_status(sector, gs, plates)
        self.fig.canvas.draw_idle()

    def _build_lattice(self, lat):
        """Static scene plus the persistent artists (tiles, arrows, labels) that
        later frames update in place instead of redrawing."""
        import matplotlib
        from matplotlib import cm, colors
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        ax = self.ax_lat
        ax.clear()
        W, H = lat.width, lat.height
        self._e_norm = colors.Normalize(-0.5, 0.5)
        self._b_norm = colors.Normalize(-1.0, 1.0)
        e_map, b_map = matplotlib.colormaps["coolwarm"], matplotlib.colormaps["PuOr_r"]

        anchors = lat.plaquette_anchors()
        self._tiles = PatchCollection(
            [Rectangle((i + 0.10, j + 0.10), 0.80, 0.80) for (i, j) in anchors],
            cmap=b_map, norm=self._b_norm, alpha=0.85, linewidths=0, zorder=0)
        self._tiles.set_array(np.zeros(len(anchors)))
        ax.add_collection(self._tiles)
        self._tile_text = [ax.text(i + 0.5, j + 0.5, "", ha="center", va="center", fontsize=8)
                           for (i, j) in anchors] if W * H <= 30 else []
        for j in (0, H - 1):                       # the plates
            ax.add_patch(Rectangle((-0.35, j - 0.13), W + 0.7, 0.26, zorder=1,
                                   color="0.25", alpha=0.35, lw=0))
            ax.text(-0.42, j, "plate", ha="right", va="center", fontsize=9, rotation=90)

        info = [lat.link_info(q) for q in range(lat.n_qubits)]
        self._link_dir = np.array([(1.0, 0.0) if d == 'x' else (0.0, 1.0) for _, _, d in info])
        origin = np.array([(float(i), float(j)) for i, j, _ in info])
        mid = origin + 0.5 * self._link_dir
        segs = np.stack([origin, origin + self._link_dir], axis=1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors="0.8", linewidths=1.0, zorder=2))
        self._arrows = ax.quiver(mid[:, 0], mid[:, 1], self._link_dir[:, 0], self._link_dir[:, 1],
                                 np.zeros(len(info)), cmap=e_map, norm=self._e_norm,
                                 pivot="mid", angles="xy", scale_units="xy", scale=1.0,
                                 width=0.007, headwidth=3.2, headlength=3.6,
                                 headaxislength=3.2, zorder=3)
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        ax.plot(xs.ravel(), ys.ravel(), 'o', ms=7, mfc="0.15", mec="0.15", zorder=4)
        ax.plot(np.full(H, W), np.arange(H), 'o', ms=7, mfc="white", mec="0.15", zorder=4)
        self._charge_text = [ax.text(i, j + off, "", ha="center", va="center", fontsize=8)
                             for (j, off) in ((0, -0.30), (H - 1, 0.30)) for i in range(W)]
        ax.set_xlim(-0.7, W + 0.4)
        ax.set_ylim(-0.6, H - 0.4)
        ax.set_aspect("equal")
        ax.set_xticks(range(W + 1))
        ax.set_xticklabels([str(i) for i in range(W)] + ["0'"])
        ax.set_yticks(range(H))
        ax.set_xlabel("x  (periodic, along the plates; 0' is the image of column 0)")
        ax.set_ylabel("y  (open, across the gap)")
        for cax, mappable, label in (
                (self.cax_e, cm.ScalarMappable(norm=self._e_norm, cmap=e_map),
                 r"electric field $\langle S^z\rangle$ (arrows)"),
                (self.cax_b, self._tiles,
                 r"magnetic energy $J\langle U_p+U_p^\dagger\rangle$ (tiles)")):
            cax.clear()
            self.fig.colorbar(mappable, cax=cax, orientation="horizontal")
            cax.set_xlabel(label, fontsize=8, labelpad=1)
            cax.tick_params(labelsize=7)
        self._scene = (W, H)

    def _draw_lattice(self, lat, efield, eplaq, charges, title, light=False):
        if not light or getattr(self, "_scene", None) != (lat.width, lat.height):
            self._build_lattice(lat)
            bmax = 1.0 if self.view == "quench" else max(0.05, abs(self.params['J']))
            self.cax_b.set_xlabel(
                r"flippability $\langle (U_p+U_p^\dagger)^2\rangle$ (tiles)" if self.view == "quench"
                else r"magnetic energy $J\langle U_p+U_p^\dagger\rangle$ (tiles)", fontsize=8, labelpad=1)
            self._b_norm.vmin, self._b_norm.vmax = -bmax, bmax
        self._tiles.set_array(eplaq)
        for txt, v in zip(self._tile_text, eplaq):
            txt.set_text(f"{v:+.2f}")
            txt.set_color("white" if abs(v) > 0.55 * self._b_norm.vmax else "0.15")
        length = np.sign(efield + 1e-15) * (0.15 + 0.6 * np.minimum(1.0, 2.0 * np.abs(efield)))
        self._arrows.set_UVC(length * self._link_dir[:, 0], length * self._link_dir[:, 1], efield)
        for txt, q in zip(self._charge_text, charges):
            txt.set_text(f"{q:+.1f}")
            txt.set_color("#b03a2e" if q > 0 else "#1f618d")
        self.ax_lat.set_title(f"{lat.width} x {lat.height} cylinder, d = {lat.height - 1}: {title}",
                              fontsize=11)

    def _draw_profile(self, lat, plaq, flip, plates, light=False):
        p = self.params
        ax = self.ax_prof
        W, H = lat.width, lat.height
        rows = np.arange(H - 1) + 0.5
        eps = (p['J'] * plaq + p['rk'] * flip).reshape(H - 1, W).mean(axis=1)
        if light and getattr(self, "_prof_line", None) is not None \
                and len(self._prof_line.get_xdata()) == len(rows):
            self._prof_line.set_ydata(eps)
            lo, hi = ax.get_ylim()
            if eps.min() < lo or eps.max() > hi:
                ax.set_ylim(min(lo, eps.min() - 0.05), max(hi, eps.max() + 0.05))
            return
        ax.clear()
        self._prof_line, = ax.plot(rows, eps, 'o-', color="#6c3483", lw=2,
                                   label="energy per plaquette, row average")
        if not light or getattr(self, "_bulk", None) is None:
            ds, E0, _ = self.model.curve(p['W'], plates, p['g'], p['J'], p['rk'], p['pattern'])
            self._bulk = ((E0[-1] - E0[-2]) / p['W'], ds[-1], ds[-2]) if len(ds) >= 2 else None
        if self._bulk is not None:
            bulk, d_hi, d_lo = self._bulk
            ax.axhline(bulk, color="k", ls="--", lw=1,
                       label=f"bulk: [E0({d_hi}) - E0({d_lo})]/W = {bulk:+.4f}")
        ax.axvspan(-0.15, 0.0, color="0.25", alpha=0.35)
        ax.axvspan(H - 1, H - 0.85, color="0.25", alpha=0.35)
        ax.set_xlim(-0.15, H - 0.85)
        ax.set_xlabel("y (plates shaded)")
        ax.set_ylabel("energy density")
        ax.set_title("(a) energy density across the gap", fontsize=10)
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(alpha=0.3)

    def _draw_casimir(self, plates):
        p = self.params
        ax, ax2 = self.ax_cas, self.ax_cas_gap
        ax.clear()
        ax2.clear()
        ds, E0, gaps = self.model.curve(p['W'], plates, p['g'], p['J'], p['rk'], p['pattern'])
        if len(ds) >= 3:
            d2 = E0[2:] - 2 * E0[1:-1] + E0[:-2]
            colours = ["#c0392b" if v < 0 else "0.6" for v in d2]
            ax.bar(ds[1:-1], d2, color=colours, width=0.6)
            ax.axhline(0, color="k", lw=0.6)
            if p['d'] in ds[1:-1]:
                ax.axvline(p['d'], color="#1f4e79", lw=1.2, ls=":")
            lim = np.max(np.abs(d2[1:])) * 1.6 if len(d2) > 1 else np.max(np.abs(d2)) * 1.2
            ax.set_ylim(-max(lim, 1e-6), max(lim, 1e-6))
        ax2.plot(ds, gaps, 's-', color="#1e8449", ms=4, lw=1)
        ax2.set_ylabel("gap", color="#1e8449", fontsize=8)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.tick_params(labelsize=7, colors="#1e8449")
        ax.set_xlabel("plate separation d")
        ax.set_ylabel(r"$D_2(d)$")
        ax.set_title("(b) Casimir signal: second difference of $E_0(d)$ "
                     "(red = attractive) and gap", fontsize=10)
        ax.grid(alpha=0.3)

    def _draw_dynamics(self, gs, light=False):
        ax = self.ax_dyn
        if self.view == "quench" and self.trace['t']:
            t = self.trace['t']
            if not (light and getattr(self, "_dyn_lines", None)):
                ax.clear()
                if getattr(self, "_ax_echo", None) is None:
                    self._ax_echo = ax.twinx()
                ax2 = self._ax_echo
                ax2.clear()
                hb, = ax.plot(t, self.trace['HB'], color="#6c3483",
                              label="mean flippability")
                ax.axhline(self._gs_flip, color="k", ls="--", lw=0.8, label="ground-state value")
                echo, = ax2.plot(t, self.trace['echo'], color="#ca6f1e", lw=1)
                ax2.set_ylim(0, 1.05)
                ax2.set_ylabel("return probability", color="#ca6f1e", fontsize=8)
                ax2.yaxis.set_label_position("right")
                ax2.yaxis.tick_right()
                ax2.tick_params(labelsize=7, colors="#ca6f1e")
                ax.legend(fontsize=7.5, loc="upper right")
                ax.set_xlabel("time")
                ax.set_title("(c) real-time dynamics", fontsize=10)
                self._dyn_lines = (hb, echo)
            else:
                self._dyn_lines[0].set_data(t, self.trace['HB'])
                self._dyn_lines[1].set_data(t, self.trace['echo'])
            ax.set_xlim(0, max(5.0, t[-1] * 1.05))
            ax.set_ylim(0, 1.05)
            return
        self._dyn_lines = None
        ax.clear()
        if getattr(self, "_ax_echo", None) is not None:
            self._ax_echo.clear()
            self._ax_echo.set_yticks([])
        ax.text(0.5, 0.5, 'choose view "quench" and press Play\n'
                "to watch vacuum fluctuations build up between the plates",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.35")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("(c) real-time dynamics", fontsize=10)

    def _draw_status(self, sector, gs, plates):
        p, lat = self.params, sector.lattice
        note = ""
        if p['g'] > 0 and p['pattern'] == "uniform":
            note = "\nuniform field commutes with H_B: it only selects a flux sector"
        if p['plates'] == "fixed" and p['W'] % 2:
            note += "\nodd W: staggered plate charges impossible, using free plates"
        self.status.set_text(
            f"link qubits N_q = {lat.n_qubits}   (+1 ancilla for RQSVT = {lat.n_qubits + 1})\n"
            f"Hilbert space 2^{lat.n_qubits} = {2 ** lat.n_qubits:.3g}   "
            f"gauge sector {sector.dim}   ground-state sector {len(gs['support'])}\n"
            f"plates: {plates}   plaquette sublayers: {lat.n_plaquette_colours}\n"
            f"E0 = {gs['energy']:+.6f}   in-sector gap = {gs['gap']:.4f}   "
            f"E0 / plaquette = {gs['energy'] / lat.n_plaquettes:+.4f}{note}")

    def save(self, path):
        self.fig.savefig(path, dpi=150)
        return path


def launch(**kwargs):
    import matplotlib.pyplot as plt
    explorer = CasimirExplorer(**kwargs)
    plt.show()
    return explorer


if __name__ == "__main__":
    launch()
