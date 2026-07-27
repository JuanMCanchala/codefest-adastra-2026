"""Definicion UNICA de los temas del corpus de prueba.

Cada entrada ata tres cosas que deben ir sincronizadas:
  - la consulta a la API de arXiv (que actua como juicio de relevancia externo),
  - la consulta equivalente en lenguaje natural (la que evalua el sistema),
  - el idioma de esa consulta.

El idioma se reparte a proposito entre ES/EN/PT y NO coincide con el idioma de
los documentos (casi todos en ingles): asi cada consulta pone a prueba la
recuperacion cross-lingual, que es el escenario real del reto (Seccion 10.1).

fetch_corpus.py descarga por `arxiv` y make_eval_corpus.py construye el eval set
por `natural`; ambos leen de aqui para no desincronizarse.
"""
from __future__ import annotations

# fenomeno -> [(consulta_arxiv, consulta_natural, idioma)]
TOPICS: dict[int, list[tuple[str, str, str]]] = {
    # ---------------- Fenomeno 1: IA en defensa y entornos militares -------
    1: [
        ('all:"autonomous weapons" AND cat:cs.CY',
         "riesgos y gobernanza de las armas autonomas letales", "es"),
        ('all:"military artificial intelligence"',
         "military applications of artificial intelligence in defense", "en"),
        ('all:"AI governance" AND cat:cs.CY',
         "governanca e regulacao da inteligencia artificial", "pt"),
        ('all:"defense" AND all:"machine learning" AND cat:cs.CY',
         "aprendizaje automatico aplicado al sector defensa", "es"),
        ('all:"lethal autonomous weapons"',
         "control humano significativo sobre el uso de la fuerza", "es"),
        ('all:"drone" AND all:"swarm" AND cat:cs.RO',
         "coordination and control of drone swarms", "en"),
        ('all:"AI safety" AND all:"policy"',
         "politicas de seguranca para sistemas de inteligencia artificial", "pt"),
        ('all:"dual-use" AND all:"technology"',
         "tecnologias de doble uso civil y militar", "es"),
        ('all:"cybersecurity" AND all:"critical infrastructure"',
         "cybersecurity threats to critical infrastructure", "en"),
        ('all:"surveillance" AND all:"ethics" AND cat:cs.CY',
         "implicacoes eticas da vigilancia automatizada", "pt"),
        ('all:"arms control" AND all:"verification"',
         "verificacion y control de armamentos", "es"),
        ('all:"human oversight" AND all:"automated decision"',
         "human oversight of automated decision systems", "en"),
        ('all:"national security" AND all:"artificial intelligence"',
         "inteligencia artificial y seguridad nacional", "es"),
    ],
    # ---------------- Fenomeno 2: Seguridad espacial y LEO -----------------
    2: [
        ('all:"space debris"',
         "desechos espaciales y basura orbital", "es"),
        ('all:"low earth orbit" AND all:"collision"',
         "collision risk between objects in low earth orbit", "en"),
        ('all:"satellite constellation" AND all:"sustainability"',
         "sustentabilidade das constelacoes de satelites", "pt"),
        ('all:"orbital debris" AND all:"mitigation"',
         "mitigacion y remocion de desechos orbitales", "es"),
        ('all:"Kessler syndrome"',
         "colisiones en cascada que degradan el entorno orbital", "es"),
        ('all:"space traffic management"',
         "coordination of traffic and operations in orbit", "en"),
        ('all:"deorbit" AND all:"satellite"',
         "descarte de satelites no fim da vida util", "pt"),
        ('all:"space situational awareness"',
         "vigilancia y seguimiento de objetos en el espacio", "es"),
        ('all:"conjunction assessment" AND all:"satellite"',
         "predicting close approaches between satellites", "en"),
        ('all:"megaconstellation" AND all:"astronomy"',
         "impacto das megaconstelacoes na observacao astronomica", "pt"),
        ('all:"space sustainability" AND all:"policy"',
         "politicas para la sostenibilidad del entorno espacial", "es"),
        ('all:"reentry" AND all:"rocket body"',
         "uncontrolled reentry of rocket stages", "en"),
        ('all:"active debris removal"',
         "mision de captura y remocion activa de desechos", "es"),
    ],
    # ---------------- Fenomeno 3: Dinamicas territoriales en LatAm ---------
    3: [
        ('all:"Latin America" AND all:"inequality"',
         "desigualdad social en America Latina", "es"),
        ('all:"Latin America" AND all:"violence"',
         "violence and organized crime in Latin America", "en"),
        ('all:"migration" AND all:"Latin America"',
         "migracao e deslocamento na America Latina", "pt"),
        ('all:"governance" AND all:"Latin America"',
         "gobernanza institucional y politicas publicas en America Latina", "es"),
        ('all:"Colombia" AND all:"conflict"',
         "dinamicas del conflicto armado en Colombia", "es"),
        ('all:"organized crime" AND all:"drug trafficking"',
         "organized crime networks and drug trafficking", "en"),
        ('all:"deforestation" AND all:"Amazon"',
         "desmatamento e degradacao ambiental na Amazonia", "pt"),
        ('all:"poverty" AND all:"Latin America"',
         "pobreza y desarrollo humano en la region", "es"),
        ('all:"education" AND all:"inequality" AND all:"Latin America"',
         "educational inequality and access to schooling", "en"),
        ('all:"urban violence" AND all:"Brazil"',
         "violencia urbana e seguranca publica nas cidades", "pt"),
        ('all:"social protest" AND all:"Latin America"',
         "protesta social y percepcion de las instituciones", "es"),
        ('all:"informal economy" AND all:"labor"',
         "informal labor markets and employment", "en"),
        ('all:"climate change" AND all:"Latin America"',
         "efectos del cambio climatico en America Latina", "es"),
    ],
}


def arxiv_queries() -> dict[int, list[str]]:
    """{fenomeno: [consultas_arxiv]} para la descarga."""
    return {fen: [t[0] for t in temas] for fen, temas in TOPICS.items()}


def natural_by_arxiv() -> dict[str, tuple[str, str]]:
    """{consulta_arxiv: (consulta_natural, idioma)} para el eval set."""
    return {t[0]: (t[1], t[2]) for temas in TOPICS.values() for t in temas}


def total_topics() -> int:
    return sum(len(v) for v in TOPICS.values())
