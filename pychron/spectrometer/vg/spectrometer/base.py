# ===============================================================================
# Copyright 2021 Stephen Cox
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
from traits.api import HasTraits, Instance
# ============= standard library imports ========================
# ============= local library imports  ==========================
from pychron.spectrometer.base_spectrometer import BaseSpectrometer
from pychron.spectrometer.vg.spectrometer.analyzer import SystemAnalyzer
from pychron.spectrometer.vg.detector.vg5400 import VG5400Detector
from pychron.spectrometer.vg.magnet.vg5400 import VG5400Magnet
from pychron.spectrometer.vg.source.vg5400 import VG5400Source

class VGSpectrometer(BaseSpectrometer):
    analyzer = Instance(SystemAnalyzer)
    detector = Instance(VG5400Detector)
    magnet = Instance(VG5400Magnet)
    source = Instance(VG5400Source)

    def load(self):
        self.detector.load()
        self.magnet.load()

    def finish_loading(self):
        self.detector.finish_loading()
        self.magnet.finish_loading()
    #
    # def _channel_select_default(self):
    #     return VG5400Detector(name='VG5400Detector',
    #                          configuration_dir_name='VG')
    #
    # def _magnet_default(self):
    #     return VG5400Magnet()
