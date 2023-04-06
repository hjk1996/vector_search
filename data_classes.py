from dataclasses import dataclass, field
import datetime


@dataclass
class Summary:
    difficulty: str = field(repr=True)
    max_lines: int = field(repr=True)
    summary: str = field(repr=True)

    @classmethod
    def from_json(cls, json):
        return cls(
            difficulty=json["difficulty"],
            max_lines=json["max_lines"],
            summary=json["summary"],
        )

    def to_json(self):
        return {
            "difficulty": self.difficulty,
            "max_lines": self.max_lines,
            "summary": self.summary,
        }


@dataclass
class ChapterSummary:
    book: str = field(repr=True)
    book_number: int = field(repr=True)
    chapter: int = field(repr=True)
    ten_line_summaries: list[Summary] = field(repr=True)
    five_line_summaries: list[Summary] = field(repr=True)
    one_line_summaries: list[Summary] = field(repr=True)

    @classmethod
    def from_json(cls, json):
        return cls(
            book=json["book"],
            book_number=json["book_number"],
            chapter=json["chapter"],
            ten_line_summaries=list(
                map(lambda x: Summary.from_json(x), json["10_line_summaries"])
            ),
            five_line_summaries=list(
                map(lambda x: Summary.from_json(x), json["5_line_summaries"])
            ),
            one_line_summaries=list(
                map(lambda x: Summary.from_json(x), json["1_line_summaries"])
            ),
        )

    def to_json(self):
        return {
            "book": self.book,
            "book_number": self.book_number,
            "chapter": self.chapter,
            "ten_line_summaries": list(
                map(lambda x: x.to_json(), self.ten_line_summaries)
            ),
            "five_line_summaries": list(
                map(lambda x: x.to_json(), self.five_line_summaries)
            ),
            "one_line_summaries": list(
                map(lambda x: x.to_json(), self.one_line_summaries)
            ),
        }

    def get_summaries(self) -> list[Summary]:
        attrs = [attr for attr in dir(self) if "summaries" in attr]
        summaries = []
        for attr in attrs:
            summaries += getattr(self, attr)
        return summaries


@dataclass
class NewsArticle:
    index: int = field(repr=True)
    title: str = field(
        repr=True,
    )
    content: str = field(repr=False)
    summaries: list[str] = field(repr=False, default=None)
    date: datetime.date = field(repr=True, default=None)

    @classmethod
    def from_json(cls, i, json):
        return cls(
            index=i,
            title=json["title"],
            content=json["content"],
            summaries=json.get("summaries", None),
            date=datetime.datetime.strptime(json["date"], "%Y-%m-%d").date()
            if json.get("date", None) != None
            else None,
        )

    def to_json(self):
        return {
            "index": self.index,
            "title": self.title,
            "content": self.content,
            "summaries": self.summaries,
            "date": self.date.strftime("%Y-%m-%d") if self.date != None else None,
        }

@dataclass
class TrainData:
    anchor: str = field(repr=True)
    pos: str = field(repr=True)
    neg: str = field(repr=True)
    book_index: int = field(repr=True)
    chapter_index: int = field(repr=True)