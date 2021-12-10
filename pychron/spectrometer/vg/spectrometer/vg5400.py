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
import time
from datetime import datetime
# ============= enthought library imports =======================
from __future__ import absolute_import

from pychron.pychron_constants import VG_DEFAULT_INTEGRATION_TIME, VG_INTEGRATION_TIMES
from pychron.spectrometer.vg.spectrometer.analyzer import SystemAnalyzer
from pychron.spectrometer.vg.detector.vg5400 import VG5400Detector
from pychron.spectrometer.vg.magnet.vg5400 import VG5400Magnet
from pychron.spectrometer.vg.source.vg5400 import VG5400Source
from pychron.spectrometer.vg.spectrometer.base import VGSpectrometer


class VG5400Spectrometer(VGSpectrometer):

    integration_times = List(ISOTOPX_INTEGRATION_TIMES)

    analyzer_klass = SystemAnalyzer
    detector_klass = VG5400Detector
    magnet_klass = VG5400Magnet
    source_klass = VG5400Source

    _read_enabled = True

    def make_configuration_dict(self):
        return {}

    def make_gains_dict(self):
        return {}

    def make_deflection_dict(self):
        return {}

    def finish_loading(self):
        super(VG5400Spectrometer, self).finish_loading()
        config = self._get_cached_config()
        if config is not None:
            magnet = config['magnet']
            mftable_name = magnet.get('mftable')
            if mftable_name:
                self.debug('updating mftable name {}'.format(mftable_name))
                self.magnet.field_table.path = mftable_name
                self.magnet.field_table.load_table(load_items=True)
            dac_ranges = magnet.get('num_ranges')

            self._generate_dac_ranges(magnet, dac_ranges)

    def _generate_dac_ranges(self, magnet, dac_ranges):
        if dac_ranges:
            self.debug('{} ranges defined for magnet DAC'.format(dac_ranges))
            for i in range(4):
                if i < dac_ranges:
                    a = magnet.get('a{}'.format(i + 1))
                    b = magnet.get('b{}'.format(i + 1))
                    c = magnet.get('c{}'.format(i + 1))

                    d_low = b ** 2 - 4 * a * c
                    d_high = b ** 2 - 4 * a * (c + 2 ** 16)

                    x1_low = (-b - d_low ** 0.5) / (2 * a)
                    x2_low = (-b + d_low ** 0.5) / (2 * a)

                    x1_high = (-b - d_high ** 0.5) / (2 * a)
                    x2_high = (-b + d_high ** 0.5) / (2 * a)

                    low = max(x1_low, x2_low)
                    high = max(x1_high, x2_high)
                    self.magnet.ranges.append([low, high, a, b, c])
                else:
                    self.magnet.ranges.append([8, 8, 0, 0, 0])
        else:
            self.debug('Warning: no ranges defined for magnet DAC. Using linear equation for full range.')
            self.magnet.ranges.append([0, 6, 0, 10922, 0])

    def start(self):
        self.set_integration_time(1, force=True)

    def read_intensities(self, timeout=60, trigger=False, detector='Faraday', verbose=False):
        self._read_enabled = True
        verbose = True

        if verbose:
            self.debug('read intensities {}'.format(detector))
        resp = True
        if trigger:
            resp = self.trigger_acq()
            if resp is not None:
                time.sleep(self.integration_time)

        keys = []
        signals = []
        collection_time = None
        inc = False

        if resp is not None:
            keys = self.detector_names[::-1]
            while self._read_enabled:
                line = self.readline(detector=detector, verbose=True)

                if verbose:
                    self.debug('raw: {}'.format(line))

                if line is None:
                    break

                if line and detector=='Faraday':
                    try:
                        args = line.split()

                        collection_time = datetime.now()

                        keys = ['F']

                        signals = [float(args[1])]

                    except BaseException as e:
                        self.debug('read intensities error={}'.format(e))

                if line and detector=='ICM':
                    try:
                        args = line.split()

                        collection_time = datetime.now()

                        keys = ['ICM']

                        signals = [float(args[1])]

                    except BaseException as e:
                        self.debug('read intensities error={}'.format(e))

        if verbose:
            self.debug('collection time: {}'.format(collection_time))
            self.debug('keys: {}'.format(keys))
            self.debug('signals: {}'.format(signals))

        return keys, signals, collection_time, True

    def readline(self, detector='Faraday', verbose=False):
        if verbose:
            self.debug('readline')
        st = time.time()
        ds = ''
        while 1:
            if time.time() - st > 3 * self.integration_time:
                if verbose:
                    self.debug('readline timeout. raw={}'.format(ds))
                return

            try:
                ds += self.read(detector=detector)
            except BaseException:
                self.debug_exception()
                self.debug(f'data left: {ds}')

            if '#\r\n' in ds:
                ds = ds.split('#\r\n')[0]
                return ds

    def get_update_period(self, it=None, is_scan=False):

    def read_integration_time(self):
        return self.integration_time

    def set_integration_time(self, it, force=False):

    def _get_simulation_data(self):
        signals = [1, 100]  # + random(6)
        keys = ['F', 'ICM']
        return keys, signals, None

    def _integration_time_default(self):
        self.default_integration_time = VG_DEFAULT_INTEGRATION_TIME
        return VG_DEFAULT_INTEGRATION_TIME