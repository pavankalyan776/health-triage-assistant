import uuid
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import HealthAction, HealthObservation, HealthState
except ImportError:
    from models import HealthAction, HealthObservation, HealthState

class HealthTriageEnvironment(Environment):
    def __init__(self):
        self._state = HealthState(episode_id=str(uuid.uuid4()), step_count=0)
        self.patient_record = "Patient: John Doe. Vitals: Temp 103F, BP 140/90. Symptoms: Chest pain."
        self.current_stage = "prioritization"

    def reset(self) -> HealthObservation:
        self._state = HealthState(episode_id=str(uuid.uuid4()), step_count=0)
        self.current_stage = "prioritization"
        return HealthObservation(
            patient_record=self.patient_record,
            current_stage=self.current_stage,
            message="New patient admitted. Assign priority.",
            reward=0.0,
            done=False
        )

    def step(self, action: HealthAction) -> HealthObservation:
        self._state.step_count += 1
        reward = 0.0
        message = ""

        # Logic for Task 1: Matches YAML ID 'prioritize_urgent_cases'
        if action.action_type == "prioritize_urgent_cases" or action.action_type == "prioritize":
            if action.value.lower() == "urgent":
                reward = 0.5
                self.current_stage = "extraction"
                message = "Correct. Patient prioritized. Now extract the temperature."

        # Logic for Task 2: Matches YAML ID 'extract_patient_vitals'
        elif action.action_type == "extract_patient_vitals" or action.action_type == "extract_vitals":
            if "103" in action.value:
                reward = 0.5
                self.current_stage = "completed"
                message = "Temperature 103F confirmed. Case triaged."

    @property
    def state(self) -> HealthState:
        return self._state