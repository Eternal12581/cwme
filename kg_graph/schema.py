from pydantic import *
from typing import Tuple, List, Dict, Optional

class ActionPrediction(BaseModel):
    query: str | None = Field(description="Include a new query if more information is needed, otherwise set to null.")
    predicted_action: str | None = Field(description="Provide the predicted action in `<action> <object>` format if no query is pending, otherwise set to null.")
    predicted_response: str | None = Field(description="Describe the expected outcome from the lastest.")
    reasoning: str | None = Field(description="Explain the rationale behind the action or query.")

class Entities(BaseModel):
    entities: Optional[List[Tuple[str, str]]] = Field(description="List of entities and their types such as location, object, person etc.")
    lemmatized_input: str

class StateDetails(BaseModel):
    observation: str = Field(description="Describe the anticipated environment post-action, using precise but concise language. Query KGQA for details if needed.")
    inventory: str | List[str] = Field(description="Predict any inventory changes resulting from your action.")
    valid_receptacles: List[str] | str = Field(description="Identify actionable elements in your observation. Base this strictly on obtainable KGQA data.")

class StateWithQuery(BaseModel):
    query: str | None = Field(description="[Optional] Pose a question for further clarity on action outcomes or environment details, or set to `null` if sufficient.")
    # predicted_response: str | None = Field(description="Describe the expected outcome from the lastest.")
    state: StateDetails | None = Field(description="`null` if `query` is used wait for the answer before predicting")
    task_completion: bool | None = Field(description="Boolean indicating if the task concludes with this state transition.")

class Relations(BaseModel):
    current_loc: str | None = Field(description="Current location of the observer, if relevant.")
    rels: List[List[str]] = Field(description="List of relationship tuples describing interactions or observations in the environment.")

class Replanning(BaseModel):
    should_replan: bool = Field(description="Indicates if replanning is necessary based on the latest outcomes.")
    reflection: str = Field(description="Analysis of the last action and its outcomes, suggesting next steps.")
    updated_subtask: str = Field(description="Proposed new subtask or direction based on the reflection and current environment status.")
    
class Actor(BaseModel):
    actions: Optional[List[str]] = None
    responses: Optional[List[str]] = None
    next_states: List[StateDetails] = None
