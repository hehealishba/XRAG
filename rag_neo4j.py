import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from rich import print
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    offer: Dict[str, Any]
    score: float


class Neo4jExplainableRAG:
    """
    Minimal, explainable RAG over a Neo4j graph using TF-IDF similarity.
    Answers are template-based to avoid hallucination and always cite sources.
    """

    def __init__(self, data_path: Path = Path("data/financial_autoleasing.jsonl")) -> None:
        load_dotenv()
        self.data_path = data_path
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "please_set_password")
        self.driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None
        self.docs: List[Dict[str, Any]] = []

    # ---------- Ingestion ----------

    def seed_graph(self) -> None:
        """Load dataset and write structured nodes plus text docs into Neo4j."""
        records = [json.loads(line) for line in self.data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with self.driver.session() as session:
            session.execute_write(self._create_constraints)
            for rec in records:
                session.execute_write(self._upsert_offer, rec)
        print(f"[green]Seeded {len(records)} offers into Neo4j[/green]")

    @staticmethod
    def _create_constraints(tx) -> None:
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Offer) REQUIRE o.id IS UNIQUE")
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Segment) REQUIRE s.name IS UNIQUE")
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.type IS UNIQUE")

    @staticmethod
    def _upsert_offer(tx, rec: Dict[str, Any]) -> None:
        doc_text = (
            f"{rec['title']} for {rec['lessee_segment']} using {rec['vehicle_type']}. "
            f"Term {rec['lease_term_months']} months, {rec['annual_mileage']} miles/year. "
            f"Base APR {rec['base_rate_apr']}%, residual {rec['residual_value_pct']}%. "
            f"Early termination: {rec['early_termination_policy']}. "
            f"Risk rules: {', '.join(rec['risk_rules'])}. "
            f"Pricing adjusters: {', '.join(rec['pricing_adjusters'])}. "
            f"FAQ: {rec['faq']}"
        )
        tx.run(
            """
            MERGE (o:Offer {id: $id})
            SET o += $offerProps
            MERGE (d:Document {id: $id})
            SET d.title = $title, d.text = $docText, d.source = $source
            MERGE (s:Segment {name: $segment})
            MERGE (v:Vehicle {type: $vehicle})
            MERGE (p:Policy {name: $earlyPolicy})
            MERGE (c:CreditModel {name: $creditModel})
            MERGE (o)-[:HAS_DOC]->(d)
            MERGE (o)-[:FOR_SEGMENT]->(s)
            MERGE (o)-[:FOR_VEHICLE]->(v)
            MERGE (o)-[:EARLY_TERMINATION]->(p)
            MERGE (o)-[:USES_CREDIT_MODEL]->(c)
            WITH o
            UNWIND $riskRules AS rule
            MERGE (r:RiskRule {rule: rule})
            MERGE (o)-[:HAS_RISK_RULE]->(r)
            WITH o
            UNWIND $pricingAdjusters AS adj
            MERGE (pa:PricingAdjuster {rule: adj})
            MERGE (o)-[:HAS_PRICING_ADJUSTER]->(pa)
            """,
            {
                "id": rec["id"],
                "offerProps": {
                    "title": rec["title"],
                    "lessee_segment": rec["lessee_segment"],
                    "vehicle_type": rec["vehicle_type"],
                    "lease_term_months": rec["lease_term_months"],
                    "annual_mileage": rec["annual_mileage"],
                    "base_rate_apr": rec["base_rate_apr"],
                    "residual_value_pct": rec["residual_value_pct"],
                    "insurance_included": rec["insurance_included"],
                    "maintenance_included": rec["maintenance_included"],
                    "early_termination_policy": rec["early_termination_policy"],
                    "credit_model": rec["credit_model"],
                    "risk_rules": rec["risk_rules"],
                    "pricing_adjusters": rec["pricing_adjusters"],
                    "faq": rec["faq"],
                    "source": rec["source"],
                },
                "title": rec["title"],
                "docText": doc_text,
                "source": rec["source"],
                "segment": rec["lessee_segment"],
                "vehicle": rec["vehicle_type"],
                "earlyPolicy": rec["early_termination_policy"],
                "creditModel": rec["credit_model"],
                "riskRules": rec["risk_rules"],
                "pricingAdjusters": rec["pricing_adjusters"],
            },
        )

    # ---------- Retrieval ----------

    def _load_documents(self) -> None:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (o:Offer)-[:HAS_DOC]->(d:Document)
                RETURN d.id AS id, d.title AS title, d.text AS text, o AS offer
                """
            )
            self.docs = [dict(record) for record in result]
        if not self.docs:
            raise RuntimeError("No documents found. Run seed_graph() first.")

    def _build_vector_index(self) -> None:
        if not self.docs:
            self._load_documents()
        corpus = [doc["text"] for doc in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, question: str, k: int = 3) -> List[RetrievedDoc]:
        if self.vectorizer is None or self.doc_matrix is None:
            self._build_vector_index()
        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        ranked: List[Tuple[int, float]] = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]
        results: List[RetrievedDoc] = []
        for idx, score in ranked:
            doc = self.docs[idx]
            results.append(
                RetrievedDoc(
                    doc_id=doc["id"],
                    title=doc["title"],
                    text=doc["text"],
                    offer=dict(doc["offer"]),
                    score=float(score),
                )
            )
        return results

    # ---------- Answer generation ----------

    def answer(self, question: str, min_score: float = 0.12, max_sources: int = 1) -> Dict[str, Any]:
        retrieved = self.retrieve(question, k=3)
        retrieved = [r for r in retrieved if r.score >= min_score]
        if not retrieved:
            return {
                "question": question,
                "answer": "No matching lease knowledge found in graph. Please narrow the question.",
                "sources": [],
                "evidence": [],
            }

        top = retrieved[0]
        offer = top.offer
        risk_focus = ", ".join(offer.get("risk_rules", [])[:2]) or "risk controls recorded"
        answer_text = (
            f"Using offer '{offer['title']}' for {offer['lessee_segment']} on {offer['vehicle_type']}, "
            f"term {offer['lease_term_months']} months at {offer['base_rate_apr']}% APR "
            f"with residual {offer['residual_value_pct']}%. Early termination: {offer['early_termination_policy']}. "
            f"Risk controls: {risk_focus}. Pricing levers: {', '.join(offer.get('pricing_adjusters', [])[:2])}."
        )

        explanation = self._explain_paths(offer_id=offer["id"])
        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"doc_id": d.doc_id, "title": d.title, "score": round(d.score, 4), "source": d.offer.get("source")}
                for d in retrieved[:max_sources]
            ],
            "evidence": explanation,
        }

    # ---------- Explainability ----------

    def _explain_paths(self, offer_id: str) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (o:Offer {id: $offerId})-[r*1..2]-(n)
                RETURN DISTINCT labels(n) AS labels, properties(n) AS props, [rel IN r | type(rel)] AS rel_types
                LIMIT 25
                """,
                {"offerId": offer_id},
            )
            evidence = []
            for record in result:
                evidence.append(
                    {
                        "labels": record["labels"],
                        "relationship_path": record["rel_types"],
                        "properties": record["props"],
                    }
                )
            return evidence


def demo() -> None:
    rag = Neo4jExplainableRAG()
    # One-time ingestion; comment out after initial load.
    rag.seed_graph()
    sample_questions = [
        "What are the risk controls for electric van leases and early termination terms?",
        "Show the pics of cats",
        "How do cross-border truck leases handle FX risk and early termination?",
        "What credit requirements apply to retail BEV leases with loyalty rebates?",
        "How are fast-approval LCV leases priced and what are the early return rules?",
    ]
    responses = []
    for q in sample_questions:
        resp = rag.answer(q)
        responses.append(resp)
        print(json.dumps(resp, indent=2))
    # Persist all outputs so they can be reviewed without copying from console.
    Path("rag_output.json").write_text(json.dumps(responses, indent=2), encoding="utf-8")


if __name__ == "__main__":
    demo()

