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
            reward=0.1,  # Safe start
            done=False
        )

    def clamp_reward(self, value: float) -> float:
        """Guarantees reward is strictly between 0 and 1 as per Scaler requirements."""
        return max(0.05, min(0.95, value))

    def step(self, action: HealthAction) -> HealthObservation:
        self._state.step_count += 1
        reward = 0.05  # Use 0.05 instead of 0.0 (Strictly > 0)
        message = "Incorrect action."
        done = False

        if action.action_type == "prioritize_urgent_cases":
            if "urgent" in action.value.lower():
                reward = 0.95  # Use 0.95 instead of 1.0 (Strictly < 1)
                self.current_stage = "extraction"
                message = "Success."

        elif action.action_type == "extract_patient_vitals":
            if "103" in action.value:
                reward = 0.95
                self.current_stage = "referral"
                message = "Success."

        elif action.action_type == "suggest_specialist_referral":
            if "cardiology" in action.value.lower():
                reward = 0.95
                message = "Triage complete."
                done = True

        return HealthObservation(
            patient_record=self.patient_record,
            current_stage=self.current_stage,
            message=message,
            reward=reward, # This will always be 0.05 or 0.95
            done=done
        )
    
    @property
    def state(self) -> HealthState:
        return self._state
