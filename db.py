import pymongo
from pymongo.collection import Collection
from tqdm import tqdm


class BibleSummaryCollection:
    def __init__(self):
        self.collection: Collection = pymongo.MongoClient("mongodb://localhost:27017/")[
            "projects"
        ]["bible_summary"]

    def make_summaries_difficulty_unique(
        self, summary_type: str = "10_line_summaries",  difficulties: list[str] = ["easy", "normal"] 
    ) -> None:
        cursor = self.collection.find(
            {f"{summary_type}.{len(difficulties)}": {"$exists": True}}
        )
        for doc in tqdm(cursor):
            summaries = doc[summary_type]
            unique_summaries = []
            for difficulty in difficulties:
                difficulty_unique_summary = next(
                    filter(lambda s: s["difficulty"] == difficulty, summaries), None
                )
                if difficulty_unique_summary:
                    unique_summaries.append(difficulty_unique_summary)
            self.collection.update_one(
                {"_id": doc["_id"]}, {"$set": {summary_type: unique_summaries}}
            )

    
