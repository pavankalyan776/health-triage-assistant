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
            reward=0.1,  # Strict: > 0.0
            done=False
        )

    def step(self, action: HealthAction) -> HealthObservation:
        self._state.step_count += 1
        reward = 0.05  # Default strict failure reward 
        message = "Incorrect action or value."
        done = False

        # Task 1: prioritize_urgent_cases
        if action.action_type == "prioritize_urgent_cases":
            if "urgent" in action.value.lower():
                reward = 0.95  # Strict: < 1.0
                self.current_stage = "extraction"
                message = "Correct. Patient prioritized. Extract vitals."
            
        # Task 2: extract_patient_vitals
        elif action.action_type == "extract_patient_vitals":
            if "103" in action.value:
                reward = 0.95
                self.current_stage = "referral"
                message = "Correct. Temp extracted. Suggest a department."

        # Task 3: suggest_specialist_referral 
        elif action.action_type == "suggest_specialist_referral":
            if "cardiology" in action.value.lower():
                reward = 0.95
                self.current_stage = "completed"
                message = "Correct. Referring to Cardiology. Triage complete."
                done = True

        if self._state.step_count >= 5:
            done = True

        return HealthObservation(
            patient_record=self.patient_record,
            current_stage=self.current_stage,
            message=message,
            reward=reward,
            done=done
        )

    @property
    def state(self) -> HealthState:
        return self._state
