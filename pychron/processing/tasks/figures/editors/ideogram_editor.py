#===============================================================================
# Copyright 2013 Jake Ross
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#============= enthought library imports =======================
from traits.api import Instance

#============= standard library imports ========================
#============= local library imports  ==========================
from pychron.processing.analyses.file_analysis import InterpretedAgeAnalysis
from pychron.processing.tasks.figures.figure_editor import FigureEditor
from pychron.processing.plotter_options_manager import IdeogramOptionsManager
from pychron.processing.plotters.figure_container import FigureContainer


class IdeogramEditor(FigureEditor):
    plotter_options_manager = Instance(IdeogramOptionsManager, ())
    basename = 'ideo'

    def plot_interpreted_ages(self, iages):
        def construct(a):
            i=InterpretedAgeAnalysis(record_id='{} ({})'.format(a.sample,a.identifier),
                                     sample=a.sample,
                                     age=a.age,
                                     age_err=a.age_err)
            return i

        po = self.plotter_options_manager.plotter_options
        for ap in po.aux_plots:
            if ap.name.lower() not in ('ideogram', 'analysis number', 'analysis number stacked'):
                ap.use = False
                ap.enabled = False

        ans=[construct(ia) for ia in iages]
        self.analyses=ans
        self._update_analyses()
        self.dump_tool()

    def get_component(self, ans, plotter_options):
        # meta = None
        # if self.figure_model:
        #     meta = self.figure_model.dump_metadata()

        if plotter_options is None:
            pom = IdeogramOptionsManager()
            plotter_options = pom.plotter_options

        model = self.figure_model
        if not model:
            from pychron.processing.plotters.ideogram.ideogram_model import IdeogramModel

            model = IdeogramModel(plot_options=plotter_options,
                                  titles=self.titles)

        model.trait_set(plot_options=plotter_options,
                        titles=self.titles,
                        analyses=ans)

        iv = FigureContainer(model=model)

        # if meta:
        #     model.load_metadata(meta)

        return model, iv.component


#============= EOF =============================================
