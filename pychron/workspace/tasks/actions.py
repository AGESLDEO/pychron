# ===============================================================================
# Copyright 2014 Jake Ross
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
from pyface.tasks.action.task_action import TaskAction
# ============= standard library imports ========================
#============= local library imports  ==========================
from pychron.envisage.resources import icon


class NewWorkspaceAction(TaskAction):
    name = 'New Workspace'
    method = 'new_workspace'
    image = icon('add')


class OpenWorkspaceAction(TaskAction):
    name = 'Open Workspace'
    method = 'open_workspace'
    image = icon('document-open')


class CheckoutAnalysesAction(TaskAction):
    name = 'Checkout Analyses'
    method = 'checkout_analyses'
    image = icon('database_go')


class AddBranchAction(TaskAction):
    name = 'Add Branch'
    method = 'add_branch'
    image = icon('add')

#============= EOF =============================================



