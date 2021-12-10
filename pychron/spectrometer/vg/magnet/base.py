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
from traits.api import Any, List, Tuple
# ============= standard library imports ========================
import time
import numpy as np
# ============= local library imports  ==========================
from pychron.hardware import get_float
from pychron.spectrometer.base_magnet import BaseMagnet


class VGMagnet(BaseMagnet):
    """
    Abstraction for the VG Magnet
    """
    device = Any
    ranges = List([Float, Float])

    # ===============================================================================
    # positioning
    # ===============================================================================
    def set_dac(self, h, verbose=False):
        """
        set the magnet dac voltage.

        determine the range then send OFJ command ::

            dev.tell('$OFJ10513000')

        :param r: r, range [0, 1, 2, 3]
        :param h: h, hp target
        :param v: v, dac voltage [must be transmitted as five octal numbers]
        :param verbose:
        :return: bool, True if dac changed else False
        """
        dev = self.device
        if dev:
            r = self._get_range(h)
            v = self._get_dac(r, h)
            dev.tell('$OFJ{0:01d}{1:05o}00'.format(r, v), verbose=verbose)
            time.sleep(self.settling_time)

        change = v != self._dac
        self._dac = v
        self.dac_changed = True
        return change

    def _get_range(self, h, verbose=False):
        """
        look up range from hp target

        :param h: h, hp target
        :param verbose:
        """
        r = 0

        if h < min(min(self.ranges)) or h > max(max(self.ranges)):
            self.debug('Hall Probe target outside of defined DAC ranges. Using first range.')
            r = 0
        else:
            for i in range(len(self.ranges), 0, -1):
                if min(self.ranges[i]) < h < max(self.ranges[i]):
                    r = i

        return r

    def _get_dac(self, r, h, verbose=False):

        """
        look up dac from range and hp target

        :param r: r, range of magnet
        :param h: h, hp target
        :param verbose:
        """
        _, _, a, b, c = self.ranges[r]

        v = a * h ** 2 + b * h + c

        return v

    @get_float
    def read_dac(self):
        return self._dac
