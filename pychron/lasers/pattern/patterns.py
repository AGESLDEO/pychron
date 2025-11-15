# ===============================================================================
# Copyright 2011 Jake Ross
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

# ============= enthought library imports =======================
from __future__ import absolute_import

import math
import os

from chaco.api import AbstractOverlay
from numpy import (
    array,
    transpose,
    linspace,
    sin,
    pi,
    append,
    arange,
    asarray,
    diff,
    roll,
    gradient,
    sign,
    hstack,
)
from scipy import signal
from six.moves import zip
from traits.api import (
    Bool,
    Float,
    Button,
    Instance,
    Range,
    Str,
    Property,
    Enum,
    on_trait_change,
)
from traits.has_traits import HasTraits
from traitsui.api import (
    View,
    Item,
    Group,
    HGroup,
    RangeEditor,
    spring,
    VGroup,
    Tabbed,
    UItem,
)

from pychron.graph.graph import Graph
from pychron.lasers.pattern.pattern_generators import circular_contour_pattern
from pychron.pychron_constants import NULL_STR
from .pattern_generators import (
    square_spiral_pattern,
    line_spiral_pattern,
    random_pattern,
    polygon_pattern,
    arc_pattern,
    line_pattern,
    trough_pattern,
    rubberband_pattern,
    raster_rubberband_pattern,
)

POLYGONS = [
    "triangle",
    "diamond",
    "pentagon",
    "hexagon",
    "heptagon",
    "octogon",
    "nonagon",
    "decagon",
]


class DirectionOverlay(AbstractOverlay):
    def overlay(self, other_component, gc, view_bounds=None, mode="normal"):
        with gc:
            gc.clip_to_rect(
                other_component.x,
                other_component.y,
                other_component.width,
                other_component.height,
            )
            a, b = self.component.map_screen(
                [(0, 0), (self.olength / 2.0, self.owidth)]
            )
            l, w = b[0] - a[0], b[1] - a[1]
            ox, oy = self.component.map_screen([(0, 0)])[0]

            gc.translate_ctm(ox, oy)
            gc.rotate_ctm(self.rotation)
            gc.translate_ctm(-ox, -oy)
            gc.translate_ctm(ox + l, oy)

            # draw 1-2
            self._draw_indicator(gc, False)

            if self.use_x:
                theta = abs(math.atan(w / (2.0 * l)))
                o = l * 0.5
                # draw 2-3
                with gc:
                    gc.translate_ctm(o, 0)
                    gc.translate_ctm(l - o, 0)
                    gc.rotate_ctm(theta)
                    gc.translate_ctm(-l + o, 0)
                    self._draw_indicator(gc, True)

                # draw 4-1
                with gc:
                    gc.translate_ctm(o, -w)

                    gc.translate_ctm(l - o, 0)
                    gc.rotate_ctm(-theta)
                    gc.translate_ctm(-l + o, 0)
                    self._draw_indicator(gc, True)

                # draw 3-4
                gc.translate_ctm(0, -w)
                self._draw_indicator(gc, False)

            else:
                if w > 8:
                    # draw verticals
                    # draw 2-3
                    with gc:
                        gc.translate_ctm(l, -w / 2.0)
                        self._draw_indicator(gc, True, False)
                        # draw 4-1
                    with gc:
                        gc.translate_ctm(-l, -w / 2.0)
                        self._draw_indicator(gc, False, False)

                # draw 3-4
                gc.translate_ctm(0, -w)
                self._draw_indicator(gc, True)

    def _draw_indicator(self, gc, left_or_down, horizontal=True):
        if left_or_down:
            if horizontal:
                gc.move_to(4, 3)
                gc.line_to(0, 0)
                gc.line_to(4, -3)
            else:
                gc.move_to(-3, 4)
                gc.line_to(0, 0)
                gc.line_to(3, 4)
        else:
            if horizontal:
                gc.move_to(-4, 3)
                gc.line_to(0, 0)
                gc.line_to(-4, -3)
            else:
                gc.move_to(-3, -4)
                gc.line_to(0, 0)
                gc.line_to(3, -4)

        gc.stroke_path()


class TargetOverlay(AbstractOverlay):
    target_radius = Float
    cx = Float
    cy = Float

    def overlay(self, component, gc, *args, **kw):
        with gc:
            x, y = self.component.map_screen([(self.cx, self.cy)])[0]
            pts = self.component.map_screen([(0, 0), (self.target_radius, 0)])
            r = abs(pts[0][0] - pts[1][0])

            gc.begin_path()
            gc.arc(x, y, r, 0, 360)
            gc.stroke_path()


class OverlapOverlay(AbstractOverlay):
    beam_radius = Float(1)

    def overlay(self, component, gc, *args, **kw):
        # gc.save_state()
        with gc:
            gc.clip_to_rect(component.x, component.y, component.width, component.height)

            xs = component.index.get_data()
            ys = component.value.get_data()
            gc.set_stroke_color((0, 0, 0, 0.5))

            pts = component.map_screen([(0, 0), (self.beam_radius, 0)])
            rad = abs(pts[0][0] - pts[1][0])

            pts = component.map_screen(list(zip(xs, ys)))

            # for i, (xi, yi) in enumerate(pts):
            #     # gc.set_fill_color((0, 0, 1, 1.0 / (0.75 * i + 1) * 0.5))
            #     gc.begin_path()
            #     gc.arc(xi, yi, rad, 0, 360)
            #     gc.draw_path()
            #     i += 1

            with gc:
                gc.set_line_join(0)
                gc.set_line_width(rad * 2)

                gc.move_to(*pts[0])

                for xi, yi in pts[1:]:
                    gc.line_to(xi, yi)

                gc.line_to(*pts[0])
                gc.line_to(*pts[1])
                gc.stroke_path()
                # gc.restore_state()


AMPLITUDE_PATTERNS = (NULL_STR, "Sine", "Square", "Saw")


class Pattern(HasTraits):
    graph = Instance(Graph, (), transient=True)
    amplitude_graph = Instance(Graph, (), transient=True)
    cx = Float(transient=True)
    cy = Float(transient=True)
    target_radius = Range(0.0, 5.0, 1)

    show_overlap = Bool(False)
    beam_radius = Range(0.0, 5.0, 1)

    path = Str
    name = Property(depends_on="path")

    xbounds = (-5, 5)
    ybounds = (-5, 5)

    velocity = Float(1)
    calculated_transit_time = Float

    niterations = Range(1, 200)
    disable_at_end = Bool(False)
    xy_pattern_enabled = Bool(True)

    z_duration = Float
    power_duration = Float

    external_duration = Float

    z_period = Float(1)
    z_duty = Float
    z_min = Float
    z_max = Float(10)
    z_offset = Float(10)
    z_sample = Float
    z_func = Enum(AMPLITUDE_PATTERNS)
    z_pattern_enabled = Bool
    z_use_transit_time = Bool

    power_period = Float
    power_duty = Float
    power_min = Float(1)
    power_max = Float(10)
    power_offset = Float(10)
    power_func = Enum(AMPLITUDE_PATTERNS)
    power_sample = Float
    power_pattern_enabled = Bool
    power_use_transit_time = Bool

    def __init__(self, *args, **kw):
        super(Pattern, self).__init__(*args, **kw)

        self.z_func = "Saw"

        self.z_duration = 5
        self.z_sample = 0.5

    @property
    def kind(self):
        return self.__class__.__name__

    @property
    def power_pattern(self):
        return self.power_func != NULL_STR and self.power_pattern_enabled

    @property
    def z_pattern(self):
        return self.z_func != NULL_STR and self.z_pattern_enabled

    def generate_name(self):
        return "{}_BR{}".format(self._basename(), self.beam_radius)

    def calculate_transit_time(self):
        try:
            self.calculated_transit_time = (
                self._get_path_length() * self.niterations
            ) / self.velocity
        except ZeroDivisionError:
            pass

        if self.power_use_transit_time:
            self.power_duration = self.calculated_transit_time
        if self.z_use_transit_time:
            self.z_duration = self.calculated_transit_time

        return self.calculated_transit_time

    def set_stage_values(self, sm):
        pass

    def pattern_generator_factory(self, **kw):
        raise NotImplementedError

    def replot(self):
        self.plot()

    def replot_power_amplitude(self):
        x, y, sx, sy = self._calculate_power_series()
        self.amplitude_graph.set_data(x)
        self.amplitude_graph.set_data(y, axis=1)

        self.amplitude_graph.set_data(sx, series=2)
        self.amplitude_graph.set_data(sy, series=2, axis=1)

    def replot_z_amplitude(self):
        x, y, sx, sy = self._calculate_z_series()
        self.amplitude_graph.set_data(x, series=1)
        self.amplitude_graph.set_data(y, series=1, axis=1)
        self.amplitude_graph.set_data(sx, series=3)
        self.amplitude_graph.set_data(sy, series=3, axis=1)

    def plot(self):
        pgen_out = self.pattern_generator_factory()
        data_out = array([pt for pt in pgen_out])
        xs, ys = transpose(data_out)

        self.graph.set_data(xs)
        self.graph.set_data(ys, axis=1)
        self._plot_hook()

        return data_out[-1][0], data_out[-1][1]

    def points_factory(self):
        gen_out = self.pattern_generator_factory()
        return list(gen_out)

    def graph_view(self):
        v = View(
            UItem("graph", style="custom"), handler=self.handler_klass, title=self.name
        )
        return v

    def clear_graph(self):
        graph = self.graph
        try:
            graph.set_data([], series=1, axis=0)
            graph.set_data([], series=1, axis=1)
            graph.set_data([], series=2, axis=0)
            graph.set_data([], series=2, axis=1)
        except IndexError:
            pass

    def reset_graph(self, **kw):
        self.graph = self._graph_factory(**kw)

    def power_values(self):
        x, y, sx, sy = self._calculate_power_series()
        return sy

    def z_values(self):
        x, y, sx, sy = self._calculate_z_series()
        return sy

    # private
    def _basename(self):
        return self.kind

    def _get_name(self):
        if not self.path:
            return "New Pattern"
        return os.path.basename(self.path).split(".")[0]

    def _get_path_length(self):
        pts = self.points_factory()
        p1 = (self.cx, self.cy)
        s = 0
        for p in pts + [
            p1,
        ]:
            d = ((p1[0] - p[0]) ** 2 + (p1[1] - p[1]) ** 2) ** 0.5
            s += d
            p1 = p

        return s

    def _get_delay(self):
        return 0

    def _calculate_power_series(self):
        # x, y, sx, sy = [], [], [], []
        # if self.power_func != NULL_STR:
        #     mi = self.power_min
        #     amp = self.power_max - self.power_min
        #     x, y, sx, sy = self._calculate_series(self.power_func, self.power_period, amp, mi, self.power_offset,
        #                                           self.power_sample, self.power_duty, self.power_duration)
        # return x, y, sx, sy
        return self._calculate_series("power")

    def _calculate_z_series(self):
        # x, y, sx, sy = [], [], [], []
        # if self.z_func != NULL_STR:
        #     mi = self.z_min
        #     amp = self.z_max - mi
        #     x, y, sx, sy = self._calculate_series(self.z_func, self.z_period, amp, mi, self.z_offset,
        #                                           self.z_sample, self.z_duty, self.z_duration)
        # return x, y, sx, sy
        return self._calculate_series("z")

    def _calculate_series(self, attr):
        mi = getattr(self, "{}_min".format(attr))
        amp = getattr(self, "{}_max".format(attr)) - mi

        funcname = getattr(self, "{}_func".format(attr))
        period = getattr(self, "{}_period".format(attr))
        offset = getattr(self, "{}_offset".format(attr))
        speriod = getattr(self, "{}_sample".format(attr))
        duty = getattr(self, "{}_duty".format(attr))
        duration = getattr(self, "{}_duration".format(attr))

        x, y, sx, sy = [], [], [], []
        if funcname != NULL_STR:
            if self.xy_pattern_enabled and getattr(
                self, "{}_use_transit_time".format(attr)
            ):
                t = self.calculate_transit_time()
            else:
                t = duration

            t = t or 1
            x = linspace(0, t, 500)

            speriod = speriod or 1
            sx = arange(0, t, speriod)
            if sx[-1] < t:
                sx = append(sx, t)

            if funcname == "Sine":

                def func(xx):
                    return (mi + amp) + amp * sin(period * xx + offset)

                y = func(x)

            elif funcname == "Square":

                def func(xx):
                    return (
                        0.5
                        * amp
                        * (
                            signal.square(
                                period * xx * 2 * pi + offset, duty=duty / 100.0
                            )
                            + 1
                        )
                        + mi
                    )

                y = func(x)

                bx = asarray(diff(y), dtype=bool)
                bx = roll(bx, 1)

                sx = x[bx]
                sx = append(asarray([0]), sx)
                sx = append(sx, asarray([t]))

            elif funcname == "Saw":

                def func(xx):
                    return (
                        0.5 * amp * (signal.sawtooth(period * xx * 2 * pi + offset) + 1)
                        + mi
                    )

                y = func(x)

                asign = sign(gradient(y))
                signchange = ((roll(asign, 1) - asign) != 0).astype(bool)
                signchange[0] = False
                nsx = x[signchange]
                sx = hstack((sx, nsx))

            sy = func(sx)

        return x, y, sx, sy

    # handlers
    def _beam_radius_changed(self):
        oo = self.graph.plots[0].plots["plot0"][0].overlays[1]
        oo.beam_radius = self.beam_radius
        self.replot()

    def _show_overlap_changed(self):
        oo = self.graph.plots[0].plots["plot0"][0].overlays[1]
        oo.visible = self.show_overlap
        oo.request_redraw()

    def _target_radius_changed(self):
        self.graph.plots[0].plots["plot0"][0].overlays[
            0
        ].target_radius = self.target_radius

    @on_trait_change("z_+,p_+")
    def _handle_amplitude_change(self, obj, name, new):
        if name in ("z_sample", "p_sample") and not new:
            return

        if name.startswith("z"):
            self.replot_z_amplitude()
        else:
            self.replot_power_amplitude()

    def _anytrait_changed(self, name, new):
        if name != "calculated_transit_time":
            self.replot()
            self.calculate_transit_time()

    # factories
    def _amplitude_graph_factory(self):
        g = Graph()
        p = g.new_plot(show_legend="ul")
        p.index_range.tight_bounds = False
        p.value_range.tight_bounds = False

        x, y, spx, spy = self._calculate_power_series()
        g.new_series(x, y, type="line", color="red")
        g.set_series_label("Power")
        x, y, szx, szy = self._calculate_z_series()
        g.new_series(x, y, type="line", color="blue")
        g.set_series_label("Z")

        g.new_series(spx, spy, type="scatter", color="red")
        g.new_series(szx, szy, type="scatter", color="blue")

        # g.new_series(type='scatter', marker='circle')
        return g

    def _graph_factory(self, **kw):
        g = Graph(window_height=250, window_width=300, container_dict=dict(padding=0))
        g.new_plot(
            bounds=[250, 250], aspect_ratio=1, resizable="", padding=[30, 0, 0, 30]
        )

        cx = self.cx
        cy = self.cy
        cbx = self.xbounds
        cby = self.ybounds
        tr = self.target_radius

        g.set_x_limits(*cbx)
        g.set_y_limits(*cby)

        lp, _plot = g.new_series()
        t = TargetOverlay(component=lp, cx=cx, cy=cy, target_radius=tr)

        lp.overlays.append(t)
        overlap_overlay = OverlapOverlay(component=lp, visible=self.show_overlap)
        lp.overlays.append(overlap_overlay)

        self._graph_factory_hook(lp)

        g.new_series(type="scatter", marker="circle")
        g.new_series(type="line", color="red")
        return g

    def _plot_hook(self):
        pass

    def _graph_factory_hook(self, lp):
        pass

    # defaults
    def _amplitude_graph_default(self):
        return self._amplitude_graph_factory()

    def _graph_default(self):
        return self._graph_factory()

    # views
    def maker_group(self):
        para_grp = Group(
            self.get_parameter_group(), show_border=True, label="Parameters"
        )
        pattern_grp = VGroup(
            HGroup(
                Item(
                    "disable_at_end",
                    label="Disable at End",
                    tooltip="Disable Laser at end of patterning",
                ),
                Item("niterations", label="N. Iterations"),
            ),
            HGroup(
                Item("velocity"),
                Item(
                    "calculated_transit_time",
                    label="Time (s)",
                    style="readonly",
                    format_str="%0.1f",
                ),
            ),
            label="XY Pattern",
            show_border=True,
        )

        display_grp = Group(
            Item("target_radius"),
            Item("show_overlap"),
            Item("beam_radius", enabled_when="show_overlap"),
            show_border=True,
            label="Display",
        )

        z_grp = self._get_amplitude_group("z", "Z")
        power_grp = self._get_amplitude_group("power", "Power")

        return Tabbed(
            VGroup(
                Item("xy_pattern_enabled"),
                para_grp,
                pattern_grp,
                display_grp,
                label="Pattern",
            ),
            z_grp,
            power_grp,
        )

    def _get_amplitude_group(self, tag, label):
        grp = VGroup(
            Item("{}_pattern_enabled".format(tag), label="Enabled"),
            VGroup(
                HGroup(
                    Item("{}_use_transit_time".format(tag)),
                    Item(
                        "{}_duration".format(tag),
                        enabled_when="not {}_use_transit_time".format(tag),
                    ),
                ),
                HGroup(
                    Item("{}_period".format(tag)),
                    Item(
                        "{}_duty".format(tag),
                        visible_when='{}_func=="Square"'.format(tag),
                    ),
                ),
                Item("{}_offset".format(tag)),
                Item("{}_min".format(tag)),
                Item("{}_max".format(tag)),
                Item("{}_func".format(tag)),
                Item(
                    "{}_sample".format(tag),
                    visible_when='{}_func!="Square"'.format(tag),
                ),
                enabled_when="{}_pattern_enabled".format(tag),
            ),
            # show_border=True,
            label=label,
        )
        return grp

    def maker_view(self):
        v = View(
            HGroup(
                self.maker_group(),
                Tabbed(
                    UItem("graph", style="custom"),
                    UItem("amplitude_graph", style="custom"),
                ),
            ),
            resizable=True,
        )
        return v

    def traits_view(self):
        v = View(
            self.maker_group(),
            buttons=["OK", "Cancel"],
            title=self.name,
            resizable=True,
        )
        return v

    def get_parameter_group(self):
        raise NotImplementedError


def rubberband_pattern(cx, cy, offset, length, rotation_deg, *, steps=None, close=True):
    """
    Generate a rectangle ("box") centered at (cx, cy) whose long axis is 'length'
    along 'rotation_deg', and whose half-width on the short axis is 'offset'.
    If offset == 0, this collapses to a single centered line segment.

    Yields (x, y) points tracing the perimeter (or the center line if offset == 0).
    """
    cx = float(cx); cy = float(cy)
    offset = float(offset); length = float(length)
    theta = math.radians(float(rotation_deg))

    # Unit vectors: long-axis (u) and short-axis (n)
    ux, uy = math.cos(theta), math.sin(theta)
    nx, ny = -uy, ux

    halfL = 0.5 * length
    halfW = offset  # symmetric on both sides of the short axis

    # If no width, just return the center line
    if halfW <= 0:
        x1 = cx - halfL * ux
        y1 = cy - halfL * uy
        x2 = cx + halfL * ux
        y2 = cy + halfL * uy
        # density ~ 1 point per unit length, min 2
        npts = max(2, int(round(max(2.0, length)))) if (steps is None) else max(2, int(steps))
        for i in range(npts):
            t = i / (npts - 1) if npts > 1 else 0.0
            yield (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return

    # Rectangle corners (counterclockwise), centered at (cx, cy)
    p_ll = (cx - halfL * ux - halfW * nx, cy - halfL * uy - halfW * ny)
    p_lu = (cx - halfL * ux + halfW * nx, cy - halfL * uy + halfW * ny)
    p_ru = (cx + halfL * ux + halfW * nx, cy + halfL * uy + halfW * ny)
    p_rl = (cx + halfL * ux - halfW * nx, cy + halfL * uy - halfW * ny)

    # Choose sampling density if not supplied
    if steps is None:
        long_pts  = max(2, int(round(max(2.0, length))))
        short_pts = max(2, int(round(max(2.0, 2 * halfW))))
    else:
        perim = 2 * (length + 2 * halfW)
        long_pts  = max(2, int(round(steps * (length / perim))))
        short_pts = max(2, int(round(steps * ((2 * halfW) / perim))))
        long_pts = max(long_pts, 2)
        short_pts = max(short_pts, 2)

    def lerp(a, b, n):
        ax, ay = a; bx, by = b
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            yield (ax + t * (bx - ax), ay + t * (by - ay))

    # Trace perimeter
    for pt in lerp(p_ll, p_lu, short_pts):       # left edge
        yield pt
    for pt in list(lerp(p_lu, p_ru, long_pts))[1:]:
        yield pt
    for pt in list(lerp(p_ru, p_rl, short_pts))[1:]:
        yield pt
    edge = list(lerp(p_rl, p_ll, long_pts))[1:]
    for pt in edge:
        yield pt
    if close and edge:
        yield p_ll


class RubberbandPattern(Pattern):
    # UI parameters
    nominal_length = Range(0.0, 25.0, 15.0, mode="slider")
    offset = Range(0.0, 5.0, 0.0, mode="slider")   # half-width on the short axis
    rotation = Range(0.0, 360.0, 0.0, mode="slider")
    stage_rotation_deg = Float(0.0)

    # view bounds
    xbounds = (-25.0, 25.0)
    ybounds = (-25.0, 25.0)

    def set_stage_values(self, sm):
        """
        Read the base rotation from stage calibration and store it (degrees).
        We don't overwrite the UI rotation; we add to it later.
        """
        try:
            rot = float(sm.canvas.calibration_item.rotation)
            # Convert if the calibration provides radians (common):
            if abs(rot) <= 6.5:
                rot = math.degrees(rot)
            self.stage_rotation_deg = rot % 360.0
        except Exception:
            # keep previous value if unavailable
            pass

    def get_parameter_group(self):
        return Group(
            Item("nominal_length", label="Length"),
            Item("rotation", label="Pattern Rotation"),
            Item("offset", label="Half Width"),
            spring,
        )

    @property
    def length(self):
        return float(self.nominal_length)

    def pattern_generator_factory(self, **kw):
        # ADD the UI rotation on top of the calibration rotation
        total_rotation = (float(self.stage_rotation_deg) + float(self.rotation)) % 360.0
        return rubberband_pattern(
            float(self.cx),
            float(self.cy),
            float(self.offset),
            float(self.length),
            total_rotation,
        )


def raster_rubberband_pattern(
    cx,
    cy,
    offset,
    length,
    rotation_deg,
    raster_step,
    *,
    cross_points=None,
):
    """
    Generate a boustrophedon (zig-zag) raster inside the same oriented rectangle:
      - Long axis length = `length` along `rotation_deg`.
      - Short axis half-width = `offset` (so total width = 2*offset).
      - Each sweep traverses ACROSS the short axis, then advances along the long axis by `raster_step`,
        then traverses back across the short axis in the opposite direction, and so on.

    Yields (x, y) points along the path.
    """
    cx = float(cx); cy = float(cy)
    offset = float(offset); length = float(length)
    step = max(1e-6, float(raster_step))
    theta = math.radians(float(rotation_deg))

    # Basis vectors
    ux, uy = math.cos(theta), math.sin(theta)  # long axis
    nx, ny = -uy, ux                           # short axis

    halfL = 0.5 * length
    halfW = max(0.0, offset)

    # Degenerate cases
    if length <= 0:
        # Just return center point
        yield (cx, cy)
        return
    if halfW == 0.0:
        # Collapse to the center line along long axis
        x1 = cx - halfL * ux; y1 = cy - halfL * uy
        x2 = cx + halfL * ux; y2 = cy + halfL * uy
        npts = max(2, int(round(max(2.0, length))))
        for i in range(npts):
            t = i / (npts - 1) if npts > 1 else 0.0
            yield (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return

    # Number of lanes along the long axis
    n_lanes = int(math.floor(length / step)) + 1
    # How many samples across each short-axis sweep (endpoints included)
    if cross_points is None:
        cross_points = max(2, int(round(max(2.0, 2.0 * halfW))))  # ~1 pt per unit width
    cross_points = max(2, int(cross_points))

    def sweep_along_short(center_pos_u, direction=1):
        """
        One sweep across the short axis at a fixed long-axis position.
        direction=+1 goes from -halfW -> +halfW, direction=-1 reverses.
        """
        # Parameter s runs from -halfW to +halfW (or reverse)
        for i in range(cross_points):
            t = i / (cross_points - 1)
            s = (-halfW) + t * (2.0 * halfW)
            if direction < 0:
                s = -s
            # world coords: cx + u*center_pos_u + n*s
            x = cx + center_pos_u * ux + s * nx
            y = cy + center_pos_u * uy + s * ny
            yield (x, y)

    # Start at the long-axis minimum end (-halfL) and march to +halfL
    current_u = -halfL
    direction = +1  # first sweep goes from -W -> +W

    for lane in range(n_lanes):
        # Clamp last lane to the true end
        if lane == n_lanes - 1:
            current_u = +halfL
        # Do one sweep across the short axis
        for pt in sweep_along_short(current_u, direction):
            yield pt
        # Advance along long axis
        current_u += step
        # Flip sweep direction for next lane
        direction *= -1


class RasterRubberbandPattern(Pattern):
    """
    Same inputs/semantics as RubberbandPattern, but path rasters across the short axis
    (zig-zag), stepping along the long axis by `raster_step` each pass.
    """
    # UI parameters
    nominal_length = Range(0.0, 25.0, 15.0, mode="slider")
    offset = Range(0.0, 5.0, 0.0, mode="slider")        # half-width on the short axis
    rotation = Range(0.0, 360.0, 0.0, mode="slider")
    raster_step = Range(0.01, 10.0, 1.0, mode="slider") # step along the long axis
    stage_rotation_deg = Float(0.0)

    xbounds = (-25.0, 25.0)
    ybounds = (-25.0, 25.0)

    def set_stage_values(self, sm):
        try:
            rot = float(sm.canvas.calibration_item.rotation)
            if abs(rot) <= 6.5:
                rot = math.degrees(rot)
            self.stage_rotation_deg = rot % 360.0
        except Exception:
            pass

    def get_parameter_group(self):
        return Group(
            Item("nominal_length", label="Length"),
            Item("rotation", label="Pattern Rotation"),
            Item("offset", label="Half Width"),
            Item("raster_step", label="Step (along length)"),
            spring,
        )

    @property
    def length(self):
        return float(self.nominal_length)

    def pattern_generator_factory(self, **kw):
        total_rotation = (float(self.stage_rotation_deg) + float(self.rotation)) % 360.0
        return raster_rubberband_pattern(
            float(self.cx),
            float(self.cy),
            float(self.offset),
            float(self.length),
            total_rotation,
            float(self.raster_step),
        )


class TroughPattern(Pattern):
    width = Range(0.0, 20.0, 10, mode="slider")
    length = Range(0.0, 20.0, 10, mode="slider")
    use_x = Bool(True)
    rotation = Range(0.0, 360.0, mode="slider")

    xbounds = (-5, 25)
    ybounds = (-5, 25)

    show_direction = Bool(True)

    def set_stage_values(self, sm):
        self.rotation = sm.canvas.calibration_item.rotation

    def _plot_hook(self):
        self.dir_overlay.trait_set(
            rotation=math.radians(self.rotation),
            use_x=self.use_x,
            olength=self.length,
            owidth=self.width,
        )

    def _graph_factory_hook(self, lp):
        self.dir_overlay = o = DirectionOverlay(
            component=lp,
            visible=self.show_direction,
            olength=self.length,
            owidth=self.width,
            use_x=self.use_x,
            rotation=self.rotation,
        )
        lp.overlays.append(o)

    def pattern_generator_factory(self, **kw):
        return trough_pattern(
            self.cx, self.cy, self.length, self.width, self.rotation, self.use_x
        )

    def get_parameter_group(self):
        return Group(
            Item("length"),
            Item("width"),
            Item("rotation"),
            Item("use_x", label="Use X Pattern"),
        )


class LinearPattern(Pattern):
    length = Float
    rotation = Range(0.0, 360.0, mode="slider")
    xbounds = (-12.5, 12.5)
    ybounds = (-12.5, 12.5)
    npasses = Range(1, 100, mode="spinner")
    cx = -10

    def _get_path_length(self):
        return self.length * self.npasses

    def pattern_generator_factory(self, **kw):
        return line_pattern(self.cx, self.cy, self.length, self.rotation, self.npasses)

    def get_parameter_group(self):
        return Group(
            Item("length"),
            Item("rotation"),
            Item(
                "npasses",
                label="N. Passes",
                tooltip="Number of times to zig zag between endpoints",
            ),
        )


class RandomPattern(Pattern):
    walk_x = Float(1)
    walk_y = Float(1)
    npoints = Range(0, 50, 10)
    regenerate = Button

    def _regenerate_fired(self):
        self.plot()

    def get_parameter_group(self):
        return Group(
            "walk_x",
            "walk_y",
            "npoints",
            HGroup(spring, Item("regenerate", show_label=False)),
        )

    def pattern_generator_factory(self, **kw):
        return random_pattern(
            self.cx, self.cy, self.walk_x, self.walk_y, self.npoints, **kw
        )

    def points_factory(self):
        gen_out = self.pattern_generator_factory()
        return [pt for pt in gen_out]


class PolygonPattern(Pattern):
    nsides = Range(3, 200)
    radius = Range(0.0, 5.0, 0.5)
    rotation = Range(0.0, 360.0, 0.0)
    show_overlap = True

    # def _get_path_length(self):
    # return (self.nsides * self.radius *
    #             math.sin(math.radians(360 / self.nsides)) + 2 * self.radius)

    #     def _get_delay(self):
    #         return 0.1 * self.nsides
    def _basename(self):
        nsides = self.nsides
        if nsides < 11:
            bn = POLYGONS[nsides - 3]
        else:
            bn = "{}gon".format(nsides)
        return bn

    def get_parameter_group(self):
        return Group(
            Item("radius"),
            Item("nsides"),
            Item("rotation", editor=RangeEditor(mode="slider", low=0, high=360)),
        )

    def pattern_generator_factory(self, **kw):
        return polygon_pattern(
            self.cx, self.cy, self.radius, self.nsides, rotation=self.rotation
        )


class ArcPattern(Pattern):
    radius = Range(0.0, 1.0, 0.5)
    degrees = Range(0.0, 360.0, 90)

    def get_parameter_group(self):
        return Group(
            "radius",
            Item("degrees", editor=RangeEditor(mode="slider", low=0, high=360)),
        )

    def pattern_generator_factory(self, **kw):
        return arc_pattern(self.cx, self.cy, self.degrees, self.radius)


class CircularPattern(Pattern):
    nsteps = Range(1, 10, 2)
    radius = Range(0.01, 0.5, 0.1)
    percent_change = Range(0.01, 5.0, 0.8)

    def get_parameter_group(self):
        return Group("radius", "nsteps", "percent_change")


class SpiralPattern(CircularPattern):
    def replot(self):
        ox, oy = self.plot()
        self.plot_in(ox, oy)

    def points_factory(self):
        gen_out = self.pattern_generator_factory()
        gen_in = self.pattern_generator_factory(direction="in")
        return [pt for pt in gen_out] + [pt for pt in gen_in]

    def plot_in(self, ox, oy):
        pgen_in = self.pattern_generator_factory(
            ox=ox, oy=oy, direction="in"  # data_out[-1][0],  # data_out[-1][1],
        )
        data_in = array([pt for pt in pgen_in])

        xs, ys = transpose(data_in)


# self.graph.set_data(xs, series=1)
#        self.graph.set_data(ys, axis=1, series=1)


class LineSpiralPattern(SpiralPattern):
    step_scalar = Range(0, 20, 5)

    def get_parameter_group(self):
        g = super(LineSpiralPattern, self).get_parameter_group()
        g.content.append(Item("step_scalar"))
        return g

    def pattern_generator_factory(self, **kw):
        return line_spiral_pattern(
            self.cx,
            self.cy,
            self.radius,
            self.nsteps,
            self.percent_change,
            self.step_scalar,
            **kw
        )


class SquareSpiralPattern(SpiralPattern):
    def pattern_generator_factory(self, **kw):
        return square_spiral_pattern(
            self.cx, self.cy, self.radius, self.nsteps, self.percent_change, **kw
        )


class CircularContourPattern(CircularPattern):
    def pattern_generator_factory(self, **kw):
        return circular_contour_pattern(
            self.cx, self.cy, self.radius, self.nsteps, self.percent_change
        )


if __name__ == "__main__":
    p = PolygonPattern()
    p.configure_traits()
# ============= EOF ====================================
