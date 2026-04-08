from typing import Literal, Optional, Dict, Any
from openenv.core.env_server import Action, Observation, State

class HealthAction(Action):
    """AI Agent's decision."""
    action_type: Literal["prioritize_urgent_cases", "extract_patient_vitals", "suggest_specialist_referral"]
    value: str

class HealthObservation(Observation):
    """What the AI Agent sees."""
    patient_record: str
    current_stage: str
    message: str
    # reward and done are inherited from Observation base

class HealthState(State):
    """Metadata for the episode."""
    case_id: str = ""
    target_vitals: str = "103F"