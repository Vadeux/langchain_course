from enum import Enum


class ApprovalState(Enum):
    APPROVED = "Approved"
    PENDING = "Pending"
    REJECTED = "Rejected"


NODE_GENERATE_SUBTOPICS = "generate_subtopics"
NODE_GET_APPROVAL = "get_approval"
NODE_CONDUCT_RESEARCH = "conduct_research"
NODE_COMPILE_REPORT = "compile_report"
