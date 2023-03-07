from dataclasses import dataclass, field


@dataclass
class ChapterSummaryResult:
    difficulty: str = field(repr=True)
    max_lines: int = field(repr=True)
    summary: str = field(repr=True)
    
    @classmethod
    def from_json(cls, json):
        return cls(
            difficulty=json["difficulty"],
            max_lines=json["max_lines"],
            summary=json["summary"]
        )
    


    def to_json(self):
        return {
            "difficulty": self.difficulty,
            "max_lines": self.max_lines,
            "summary": self.summary
        }

