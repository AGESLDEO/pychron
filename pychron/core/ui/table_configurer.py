# ===============================================================================
# Copyright 2013 Jake Ross
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
import os
# ============= enthought library imports =======================
import pickle
# import apptools.sweet_pickle as pickle
# ============= standard library imports ========================
from datetime import datetime

from traits.api import HasTraits, List, Bool, Int, Instance, Enum, \
    Str, Callable, Button, Property
from traits.trait_errors import TraitError
from traitsui.api import View, Item, UItem, CheckListEditor, VGroup, Handler, HGroup, Tabbed, InstanceEditor, \
    TableEditor, EnumEditor
# ============= local library imports  ==========================
from traitsui.table_column import ObjectColumn
from traitsui.tabular_adapter import TabularAdapter

from pychron.core.helpers.traitsui_shortcuts import okcancel_view
from pychron.core.pychron_traits import BorderVGroup
from pychron.paths import paths
from pychron.pychron_constants import ARGON_KEYS, SIZES


class TableConfigurerHandler(Handler):
    def closed(self, info, is_ok):
        if is_ok:
            info.object.closed(is_ok)
            # info.object.dump()
            # info.object.set_columns()


def get_columns_group():
    col_grp = VGroup(UItem('columns',
                           style='custom',
                           editor=CheckListEditor(name='available_columns', cols=3)),
                     label='Columns', show_border=True)
    return col_grp


class TableConfigurer(HasTraits):
    columns = List
    children = List(TabularAdapter)
    available_columns = List
    sparse_columns = List
    adapter = Instance(TabularAdapter)
    id = 'table'
    font = Enum(*SIZES)
    auto_set = Bool(False)
    fontsize_enabled = Bool(True)
    title = Str('Configure Table')
    refresh_func = Callable
    show_all = Button('Show All')

    set_sparse = Button('Define Sparse')
    toggle_sparse = Button('Toggle Sparse')

    sparse_enabled = Property(depends_on='columns[]')

    _toggle_sparse_enabled = Bool(False)

    default_button = Button('Default')
    defaults_path = Str

    def _get_sparse_enabled(self):
        return self.sparse_columns != self.columns and len(self.columns) < 5

    def __init__(self, *args, **kw):
        super(TableConfigurer, self).__init__(*args, **kw)
        if self.auto_set:
            self.on_trait_change(self.update, 'font, columns[]')
        self._load_state()

    def closed(self, is_ok):
        if is_ok:
            self.dump()
            self.set_columns()
            self.set_font()

    def load(self):
        self._load_state()

    def dump(self):
        self._dump_state()

    def update(self):
        self.set_font()
        self.set_columns()

    def set_font(self):
        if self.adapter:
            font = 'arial {}'.format(self.font)
            self.adapter.font = font
            for ci in self.children:
                ci.font = font

            if self.refresh_func:
                self.refresh_func()

                # self.refresh_table_needed = True

    def set_columns(self):
        # def _columns_changed(self):
        if self.adapter:
            cols = self._assemble_columns()
            for ci in self.children:
                ci.columns = cols

            cols = [ci for ci in cols if ci in self.adapter.all_columns]
            self.adapter.columns = cols

    def _set_font(self, f):
        s = f.pointSize()
        self.font = s

    def _load_state(self):
        p = os.path.join(paths.hidden_dir, self.id)
        if os.path.isfile(p):
            try:
                with open(p, 'rb') as rfile:
                    state = pickle.load(rfile)

            except (pickle.PickleError, OSError, EOFError, TraitError):
                return

            try:
                self.sparse_columns = state.get('sparse_columns')
            except:
                pass

            cols = state.get('columns')
            if cols:
                ncols = []
                for ai in self.available_columns:
                    if ai in cols:
                        ncols.append(ai)

                self.columns = ncols

            font = state.get('font', None)
            if font:
                self.font = font

            self._load_hook(state)
            self.update()

    def _dump_state(self):
        p = os.path.join(paths.hidden_dir, self.id)
        obj = self._get_dump()

        with open(p, 'wb') as wfile:
            try:
                pickle.dump(obj, wfile)
            except pickle.PickleError:
                pass

    def _get_dump(self):
        obj = dict(columns=self.columns,
                   font=self.font,
                   sparse_columns=self.sparse_columns)
        return obj

    def _load_hook(self, state):
        pass

    def _assemble_columns(self):
        d = self.adapter.all_columns_dict
        return [(k, d[k]) for k, v in self.adapter.all_columns if k in self.columns]

    def _get_columns_grp(self):
        return

    def _set_defaults(self):
        p = self.defaults_path
        if os.path.isfile(p):
            import yaml

            with open(p, 'r') as rfile:
                yd = yaml.load(rfile)
                try:
                    self.columns = yd['columns']
                except KeyError:
                    pass
            self.set_columns()

    def _default_button_fired(self):
        self._set_defaults()

    def _set_sparse_fired(self):
        self.sparse_columns = self.columns

    def _toggle_sparse_fired(self):
        if self._toggle_sparse_enabled:
            columns = self._prev_columns
        else:
            self._prev_columns = self.columns
            columns = self.sparse_columns

        self.columns = columns
        self.set_columns()

        self._toggle_sparse_enabled = not self._toggle_sparse_enabled

    def _show_all_fired(self):
        self.columns = self.available_columns
        self.set_columns()

    def set_adapter(self, adp):
        self.adapter = adp
        # def _adapter_changed(self, adp):
        #     if adp:
        acols = [c for c, _ in adp.all_columns]

        # set currently visible columns
        t = [c for c, _ in adp.columns]

        cols = [c for c in acols if c in t]
        self.trait_set(columns=cols)

        # set all available columns
        self.available_columns = acols
        if adp.font:
            self._set_font(adp.font)

        self._load_state()

    def traits_view(self):
        v = okcancel_view(VGroup(HGroup(UItem('show_all', tooltip='Show all columns'),
                                        UItem('set_sparse',
                                              tooltip='Set the current set of columns to the Sparse Column Set',
                                              enabled_when='sparse_enabled'),
                                        UItem('toggle_sparse',
                                              tooltip='Display only Sparse Column Set'),
                                        UItem('default_button',
                                              tooltip='Set to Laboratory defaults. File located at '
                                                      '[root]/experiments/experiment_defaults.yaml')),
                                 VGroup(UItem('columns',
                                              style='custom',
                                              editor=CheckListEditor(name='available_columns', cols=3)),
                                        Item('font', enabled_when='fontsize_enabled'),
                                        show_border=True)),
                          handler=TableConfigurerHandler(),
                          title=self.title)
        return v


def str_to_time(lp):
    lp = lp.replace('/', '-')
    if lp.count('-') == 2:
        y = lp.split('-')[-1]
        y = 'y' if len(y) == 2 else 'Y'

        fmt = '%m-%d-%{}'.format(y)
    elif lp.count('-') == 1:
        y = lp.split('-')[-1]
        y = 'y' if len(y) == 2 else 'Y'

        fmt = '%m-%{}'.format(y)
    else:
        fmt = '%Y' if len(lp) == 4 else '%y'

    return datetime.strptime(lp, fmt)


class ExperimentTableConfigurer(TableConfigurer):
    defaults_path = paths.experiment_defaults


class AnalysisTableConfigurer(TableConfigurer):
    id = 'analysis.table'
    limit = Int
    omit_invalid = Bool(True)

    def _get_dump(self):
        obj = super(AnalysisTableConfigurer, self)._get_dump()
        obj['limit'] = self.limit
        obj['omit_invalid'] = self.omit_invalid

        return obj

    def _load_hook(self, obj):
        self.limit = obj.get('limit', 500)
        self.omit_invalid = obj.get('omit_invalid', True)

    def traits_view(self):
        v = okcancel_view(VGroup(get_columns_group(),
                                 Item('omit_invalid'),
                                 Item('limit',
                                      tooltip='Limit number of displayed analyses',
                                      label='Limit'),
                                 show_border=True,
                                 label='Limiting'),
                          buttons=['OK', 'Cancel', 'Revert'],
                          title=self.title,
                          handler=TableConfigurerHandler,
                          width=300)
        return v


class SampleTableConfigurer(TableConfigurer):
    title = 'Configure Sample Table'
    id = 'sample.table'
    filter_non_run_samples = Bool(True)

    def _get_dump(self):
        obj = super(SampleTableConfigurer, self)._get_dump()
        obj['filter_non_run_samples'] = self.filter_non_run_samples

        return obj

    def _load_hook(self, obj):
        self.filter_non_run_samples = obj.get('filter_non_run_samples', True)

    def traits_view(self):
        v = okcancel_view(VGroup(get_columns_group(),
                                 Item('filter_non_run_samples',
                                      tooltip='Omit samples that have not been analyzed to date',
                                      label='Exclude Non-Run')),
                          buttons=['OK', 'Cancel', 'Revert'],
                          title=self.title,
                          handler=TableConfigurerHandler,
                          width=300)
        return v


class IsotopeTableConfigurer(TableConfigurer):
    id = 'recall.isotopes'

    def traits_view(self):
        v = View(VGroup(get_columns_group(),
                        Item('font', enabled_when='fontsize_enabled'),
                        show_border=True,
                        label='Isotopes'))
        return v


class IntermediateTableConfigurer(TableConfigurer):
    id = 'recall.intermediate'


class Ratio(HasTraits):
    isotopes = List([''] + list(ARGON_KEYS))
    numerator = Str
    denominator = Str

    @property
    def tagname(self):
        if self.numerator and self.denominator:
            return '{}/{}'.format(self.numerator, self.denominator)

    def get_dump(self):
        return {'numerator': self.numerator, 'denominator': self.denominator}


class CocktailOptions(HasTraits):
    ratios = List

    def get_dump(self):
        return {'ratios': [r.get_dump() for r in self.ratios]}

    def set_ratios(self, ratios):
        self.ratios = [Ratio(numerator=r['numerator'], denominator=r['denominator']) for r in ratios]

    def _ratios_default(self):
        return [Ratio() for i in range(10)]

    def traits_view(self):
        cols = [ObjectColumn(name='numerator', editor=EnumEditor(name='isotopes')),
                ObjectColumn(name='denominator', editor=EnumEditor(name='isotopes'))]
        v = View(BorderVGroup(UItem('ratios',
                                    editor=TableEditor(sortable=False, columns=cols)),
                              label='Cocktail Options'))
        return v


class RecallOptions(HasTraits):
    cocktail_options = Instance(CocktailOptions, ())
    isotope_sig_figs = Int(5)
    computed_sig_figs = Int(5)
    intermediate_sig_figs = Int(5)

    def set_cocktail(self, co):
        cc = CocktailOptions()
        cc.set_ratios(co.get('ratios'))
        self.cocktail_options = cc

    def get_dump(self):
        return {'cocktail_options': self.cocktail_options.get_dump(),
                'computed_sig_figs': self.computed_sig_figs,
                'sig_figs': self.isotope_sig_figs,
                'intermediate_sig_figs': self.intermediate_sig_figs}

    def traits_view(self):
        v = View(Item('computed_sig_figs', label='Main SigFigs'),
                 Item('isotope_sig_figs', label='Isotope SigFigs'),
                 Item('intermediate_sig_figs', label='Intermediate SigFigs'),
                 UItem('cocktail_options', style='custom'))
        return v


class RecallTableConfigurer(TableConfigurer):
    isotope_table_configurer = Instance(IsotopeTableConfigurer, ())
    intermediate_table_configurer = Instance(IntermediateTableConfigurer, ())
    show_intermediate = Bool
    experiment_fontsize = Enum(*SIZES)
    measurement_fontsize = Enum(*SIZES)
    extraction_fontsize = Enum(*SIZES)
    main_measurement_fontsize = Enum(*SIZES)
    main_extraction_fontsize = Enum(*SIZES)
    main_computed_fontsize = Enum(*SIZES)

    subview_names = ('experiment', 'measurement', 'extraction')
    main_names = ('measurement', 'extraction', 'computed')
    bind_fontsizes = Bool(False)
    global_fontsize = Enum(*SIZES)

    recall_options = Instance(RecallOptions, ())

    def _get_dump(self):
        obj = super(RecallTableConfigurer, self)._get_dump()
        obj['show_intermediate'] = self.show_intermediate
        for a in self.subview_names:
            a = '{}_fontsize'.format(a)
            obj[a] = getattr(self, a)

        for a in self.main_names:
            a = 'main_{}_fontsize'.format(a)
            obj[a] = getattr(self, a)

        for attr in ('global_fontsize', 'bind_fontsizes'):
            obj[attr] = getattr(self, attr)

        obj['recall_options'] = self.recall_options.get_dump()
        return obj

    def _load_hook(self, obj):
        self.show_intermediate = obj.get('show_intermediate', True)
        self.isotope_table_configurer.load()
        self.intermediate_table_configurer.load()

        for a in self.subview_names:
            a = '{}_fontsize'.format(a)
            setattr(self, a, obj.get(a, 10))

        for a in self.main_names:
            a = 'main_{}_fontsize'.format(a)
            setattr(self, a, obj.get(a, 10))

        for attr in ('global_fontsize', 'bind_fontsizes'):
            try:
                setattr(self, attr, obj[attr])
            except KeyError:
                pass

        recall_options = obj.get('recall_options')
        if recall_options:
            r = RecallOptions()
            r.set_cocktail(recall_options.get('cocktail_options'))
            for tag in ('intermediate', 'isotope', 'computed'):
                tag = '{}_sig_figs'.format(tag)
                setattr(r, tag, recall_options.get(tag, 5))

            self.recall_options = r

    def dump(self):
        super(RecallTableConfigurer, self).dump()
        self.intermediate_table_configurer.dump()
        self.isotope_table_configurer.dump()

    def set_columns(self):
        self.isotope_table_configurer.set_columns()
        self.intermediate_table_configurer.set_columns()

        self.isotope_table_configurer.update()
        self.intermediate_table_configurer.update()

    def set_font(self):
        self.isotope_table_configurer.set_font()
        self.intermediate_table_configurer.set_font()

    def set_fonts(self, av):
        self.set_font()

        for a in self.subview_names:
            s = getattr(self, '{}_fontsize'.format(a))
            av.update_fontsize(a, s)

        for a in self.main_names:
            av.update_fontsize('main.{}'.format(a),
                               getattr(self, 'main_{}_fontsize'.format(a)))

        av.main_view.refresh_needed = True

    def _bind_fontsizes_changed(self, new):
        if new:
            self._global_fontsize_changed()
        self.isotope_table_configurer.fontsize_enabled = not new
        self.intermediate_table_configurer.fontsize_enabled = not new

    def _global_fontsize_changed(self):
        gf = self.global_fontsize
        self.isotope_table_configurer.font = gf
        self.intermediate_table_configurer.font = gf

        self.main_measurement_fontsize = gf
        self.main_extraction_fontsize = gf
        self.main_computed_fontsize = gf

    def traits_view(self):
        main_grp = VGroup(HGroup(Item('bind_fontsizes'),
                                 Item('global_fontsize', enabled_when='bind_fontsizes')),
                          Item('main_extraction_fontsize', enabled_when='not bind_fontsizes'),
                          Item('main_measurement_fontsize', enabled_when='not bind_fontsizes'),
                          Item('main_computed_fontsize', enabled_when='not bind_fontsizes'))

        main_view = VGroup(main_grp,
                           UItem('isotope_table_configurer', style='custom'),
                           HGroup(Item('show_intermediate', label='Show Intermediate Table')),
                           UItem('intermediate_table_configurer', style='custom', enabled_when='show_intermediate'),
                           label='Main')

        # experiment_view = VGroup(Item('experiment_fontsize', label='Size'),
        #                          show_border=True,
        #                          label='Experiment')
        # measurement_view = VGroup(Item('measurement_fontsize', label='Size'),
        #                           show_border=True,
        #                           label='Measurement')
        # extraction_view = VGroup(Item('extraction_fontsize', label='Size'),
        #                          show_border=True,
        #                          label='Extraction')

        v = View(Tabbed(main_view,
                        UItem('recall_options', editor=InstanceEditor(), style='custom'),
                        # VGroup(experiment_view,
                        #        measurement_view,
                        #        extraction_view, label='Scripts')
                        ),

                 buttons=['OK', 'Cancel', 'Revert'],
                 kind='livemodal',
                 title='Configure Table',
                 handler=TableConfigurerHandler,
                 resizable=True,
                 width=300)
        return v

# ============= EOF =============================================
