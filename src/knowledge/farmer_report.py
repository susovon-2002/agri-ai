from pathlib import Path
import json
import re


# ============================================================
# AgriVision AI - Knowledge-Based Farmer Report Generator
# ============================================================

from .wikipedia import wikipedia_report_section


BASE_DIR = Path(__file__).resolve().parents[2]

KNOWLEDGE_FILE = (
    BASE_DIR
    / "data"
    / "knowledge"
    / "disease_information.json"
)


# ============================================================
# LOAD DISEASE KNOWLEDGE DATABASE
# ============================================================

with open(
    KNOWLEDGE_FILE,
    "r",
    encoding="utf-8"
) as f:
    DISEASE_KNOWLEDGE = json.load(f)


# ============================================================
# HELPERS
# ============================================================

def _format_title(value):
    """Convert internal class names into readable text."""

    replacements = {
        "__": " ",
        "___": " ",
        "_": " ",
    }

    text = str(value)

    for old, new in replacements.items():
        text = text.replace(old, new)

    return " ".join(text.split()).strip()


def _get_value(data, key, default=None):
    value = data.get(key, default)

    if value is None:
        return default

    return value


def _format_list(items):
    """Convert strings or lists into readable Markdown."""

    if not items:
        return "- Information not available in the current knowledge database."

    # A single text value should remain one paragraph,
    # not be split into individual characters.
    if isinstance(items, str):
        return items

    # Lists/tuples are rendered as bullet points.
    if isinstance(items, (list, tuple)):
        return "\n".join(
            f"- {item}"
            for item in items
        )

    return str(items)


def _slug_to_title(slug):
    if not slug:
        return "Agricultural Source"

    slug = slug.strip().strip("/")
    slug = slug.replace("%20", " ")
    slug = slug.replace("_", " ")
    slug = slug.replace("/", " ")

    # Handle the specific agricultural links requested by the report.
    lower = slug.lower()
    if "late-blight" in lower and "tomato" in lower and "potato" in lower:
        return "Tomato & Potato Late Blight"
    if "late-blight" in lower:
        return "Late Blight Disease Management"
    if "tomato" in lower and "potato" in lower:
        return "Tomato & Potato Late Blight"
    if "tomato" in lower and "home-garden" in lower:
        return "Tomato Disease Management"
    if "disease-management" in lower:
        return "Disease Management"

    # Clean common route components and create a readable title without the raw URL.
    slug = slug.replace("-", " ")
    slug = re.sub(r"\b(?:agriculture|specialty crops|specialty-crops|vegetable farming|vegetable-farming|disease management|disease-management|home garden|home-garden)\b", "", slug, flags=re.IGNORECASE)
    slug = re.sub(r"\s+", " ", slug).strip()

    words = []
    for part in slug.split():
        if not part:
            continue
        if part.lower() in {"and", "or", "of", "in", "the", "for", "to", "a", "an", "on"}:
            words.append(part.lower())
        else:
            words.append(part.capitalize())

    title = " ".join(words)
    title = title.replace("Tomato Potato", "Tomato & Potato")
    return title or "Agricultural Source"


def _source_label(source):
    if isinstance(source, dict):
        name = str(source.get("name") or "").strip()
        url = str(source.get("url") or "").strip()
        if name:
            return name, url
        if url:
            return _source_label(url)
        return "Agricultural Source", ""

    if not isinstance(source, str):
        return str(source), ""

    candidate = source.strip()
    if not candidate:
        return "Agricultural Source", ""

    url = candidate
    parsed = candidate.split("#", 1)[0]

    if parsed.startswith("http://") or parsed.startswith("https://"):
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(parsed)
            host = parsed_url.netloc.lower()
            path = parsed_url.path.strip("/")
        except Exception:
            host = ""
            path = ""

        if "extension.umn.edu" in host:
            title = "University of Minnesota Extension"
            if path:
                title += " — " + _slug_to_title(path)
            return title, parsed

        if "extension.psu.edu" in host:
            title = "Penn State Extension"
            if path:
                title += " — " + _slug_to_title(path)
            return title, parsed

        if path:
            return _slug_to_title(path), parsed

    return candidate, url


def _format_sources(sources):
    """Return a compact evidence note instead of a list of visible source links."""
    evidence = [
        "University of Minnesota Extension",
        "Penn State Extension",
        "Wikipedia",
    ]

    if not sources:
        return (
            "Evidence used: " + "; ".join(evidence) + ". "
            "The report content reflects the local agricultural knowledge database and Wikipedia summary."
        )

    names = []
    for source in sources:
        name, _ = _source_label(source)
        if name and name not in names:
            names.append(name)

    if not names:
        names = evidence

    # Keep the report content informative and compact; do not display raw URLs.
    return (
        "Evidence used: " + "; ".join(names[:3]) + ". "
        "The disease information in this report was integrated from authoritative agricultural references and local knowledge."
    )


def _format_symptom_explanations(items):
    """Render structured symptom explanations in a readable format."""
    if not items:
        return "- Not applicable for this disease."

    if isinstance(items, str):
        return items

    if not isinstance(items, list):
        return str(items)

    lines = []
    for item in items:
        if isinstance(item, dict):
            symptom = item.get("symptom", "")
            why = item.get("why_it_occurs", "")
            when = item.get("when_it_occurs", "")
            what = item.get("what_it_indicates", "")

            if not any([symptom, why, when, what]):
                lines.append(f"- {item}")
                continue

            if symptom:
                lines.append(f"- {symptom}")
            if why:
                lines.append(f"  Why it occurs: {why}")
            if when:
                lines.append(f"  When it occurs: {when}")
            if what:
                lines.append(f"  What it indicates: {what}")
        else:
            lines.append(f"- {item}")

    return "\n".join(lines)


def _format_disease_triangle(disease_triangle):
    if not disease_triangle:
        return "- Details not available."

    if isinstance(disease_triangle, str):
        return disease_triangle

    parts = []
    for label, text in (
        ("Pathogen", disease_triangle.get("pathogen")),
        ("Host", disease_triangle.get("host")),
        ("Environment", disease_triangle.get("environment")),
    ):
        if text:
            parts.append(f"- **{label}:** {text}")

    return "\n".join(parts) if parts else "- Details not available."


def _format_section(label, value):
    if value in (None, "", [], {}):
        return ""

    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, list):
        text = _format_list(value)
    else:
        text = str(value)

    if not text:
        return ""

    return f"{text}\n"


def _safe_report_text(text):
    if text is None:
        return ""
    return str(text)


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_report(result):
    """
    Generate a detailed farmer-friendly report using:

    1. ResNet18 computer-vision result
    2. affected-leaf estimate
    3. severity estimate
    4. local disease_information.json knowledge database

    No external AI service.
    No external language model.
    No external AI API.
    """

    predicted_class = result.get(
        "predicted_class",
        result.get(
            "prediction",
            "Unknown condition"
        )
    )

    confidence = float(
        result.get(
            "confidence_percent",
            result.get(
                "confidence",
                0
            )
        )
        or 0
    )

    affected = float(
        result.get(
            "affected_leaf_percent",
            result.get(
                "affected_leaf",
                0
            )
        )
        or 0
    )

    severity = str(
        result.get(
            "severity",
            "Unknown"
        )
    )

    knowledge = DISEASE_KNOWLEDGE.get(
        predicted_class
    )

    # --------------------------------------------------------
    # Missing knowledge entry
    # --------------------------------------------------------

    if knowledge is None:

        disease_name = _format_title(
            predicted_class
        )

        return f"""
## 🌿 AgriVision AI — Farmer Report

**1. Crop / Plant**

**{disease_name}**

**2. Detected Condition**

**{disease_name}**

The computer-vision model identified this condition from the uploaded leaf image.

**3. Model Confidence**

**{confidence:.2f}%**

**4. Estimated Severity**

**{severity}**

**5. Estimated Affected Leaf Area**

**{affected:.2f}%**

**6. Important Note**

Detailed knowledge for this model class is not available in the local agricultural knowledge database.

The result is therefore limited to the computer-vision classification.

This is **not a laboratory diagnosis**.

For treatment or management decisions, consult a qualified local agricultural expert or plant pathologist.
"""

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    crop = _get_value(knowledge, "crop", "Not specified")
    condition = _get_value(knowledge, "condition", _format_title(predicted_class))
    disease_type = _get_value(knowledge, "disease_type", "Not specified")
    causal_agent = _get_value(knowledge, "causal_agent", "Not specified")
    scientific_name = _get_value(knowledge, "scientific_name", "Not specified")
    pathogen_group = _get_value(knowledge, "pathogen_group", "Not specified")
    definition = _get_value(knowledge, "definition", "Information not available.")
    detailed_description = _get_value(knowledge, "detailed_description", "")
    disease_triangle = _get_value(knowledge, "disease_triangle", {})
    why_it_occurs = _get_value(knowledge, "why_it_occurs", "")
    how_it_starts = _get_value(knowledge, "how_it_starts", [])
    disease_cycle = _get_value(knowledge, "disease_cycle", "")
    how_it_spreads = _get_value(knowledge, "how_it_spreads", [])
    favorable_conditions = _get_value(knowledge, "favorable_conditions", [])
    unfavorable_conditions = _get_value(knowledge, "unfavorable_conditions", [])
    environmental_risk = _get_value(knowledge, "environmental_risk", "")
    early_symptoms = _get_value(knowledge, "early_symptoms", [])
    symptom_explanations = _get_value(knowledge, "symptom_explanations", [])
    leaf_symptoms = _get_value(knowledge, "leaf_symptoms", [])
    leaf_symptom_explanations = _get_value(knowledge, "leaf_symptom_explanations", [])
    stem_symptoms = _get_value(knowledge, "stem_symptoms", [])
    stem_symptom_explanations = _get_value(knowledge, "stem_symptom_explanations", [])
    fruit_symptoms = _get_value(knowledge, "fruit_symptoms", [])
    fruit_symptom_explanations = _get_value(knowledge, "fruit_symptom_explanations", [])
    root_or_tuber_symptoms = _get_value(knowledge, "root_or_tuber_symptoms", [])
    root_or_tuber_symptom_explanations = _get_value(knowledge, "root_or_tuber_symptom_explanations", [])
    advanced_symptoms = _get_value(knowledge, "advanced_symptoms", [])
    disease_progression = _get_value(knowledge, "disease_progression", "")
    identification = _get_value(knowledge, "identification", [])
    similar_conditions = _get_value(knowledge, "similar_conditions", [])
    monitoring = _get_value(knowledge, "monitoring", [])
    natural_prevention = _get_value(knowledge, "natural_prevention", [])
    cultural_prevention = _get_value(knowledge, "cultural_prevention", [])
    management = _get_value(knowledge, "management", [])
    what_to_avoid = _get_value(knowledge, "what_to_avoid", [])
    weather_risk = _get_value(knowledge, "environmental_risk", "")
    when_to_seek_expert_help = _get_value(knowledge, "when_to_seek_expert_help", "")
    important_note = _get_value(knowledge, "important_note", "")
    sources = _get_value(knowledge, "sources", [])

    # Healthy classes are handled separately with a structured report that clearly states no disease was detected.
    is_healthy_class = "healthy" in str(disease_type).lower() or "healthy" in str(condition).lower()

    sections = []
    counter = 1

    def add_section(title, body):
        nonlocal counter
        body = _safe_report_text(body)
        if body in (None, "", "\n"):
            return
        sections.append(f"**{counter}. {title}**\n\n{body}\n")
        counter += 1

    # ------------------------------------------------------------------
    # Healthy plant report
    # ------------------------------------------------------------------
    if is_healthy_class:
        report = """
## 🌿 AgriVision AI — Farmer-Friendly Scientific Report

**This class is a healthy plant classification.**

No disease was detected by the model for this class.

The result is a model-based observation from the uploaded image and does not guarantee that every plant in a field is disease-free.
"""
        add_section("Crop / Plant", crop)
        add_section("Detected Condition", condition)
        add_section("Disease Type", disease_type)
        add_section("Healthy Plant Characteristics", "This image is consistent with healthy green tissue and no disease-specific lesion pattern. Healthy leaves normally show consistent color, normal leaf structure and no obvious pathogen-associated lesions.")
        add_section("Why the image is classified as healthy", "The computer-vision model selected the healthy class because the uploaded image matched the characteristics of healthy tomato, potato or pepper leaf tissue in the trained dataset.")
        add_section("Good Environmental Conditions", _format_list(favorable_conditions if favorable_conditions else ["Balanced moisture", "Adequate nutrition", "Good airflow", "Routine monitoring"]))
        add_section("Healthy Leaf Characteristics", _format_list(["Consistent green color", "Normal leaf texture", "No diagnostic pattern of disease lesions", "No obvious chlorosis, wilt or collapse"]))
        add_section("Preventive Crop Care", _format_list(natural_prevention if natural_prevention else ["Maintain healthy growing conditions", "Use healthy planting material", "Monitor crop regularly", "Keep irrigation and nutrition balanced"]))
        add_section("Monitoring", _format_list(monitoring if monitoring else ["Continue regular scouting", "Inspect leaves, stems and fruit for any new symptoms", "Check irrigation and nutrition management"]))
        add_section("Early Warning Signs to Watch For", _format_list(["Any new yellowing", "Lesion development", "Wilting", "Visible pest feeding", "Stunting or deformation"]))
        add_section("When to Seek Agricultural Advice", when_to_seek_expert_help if when_to_seek_expert_help else "Seek advice if any new lesions, wilting, yellowing, pest feeding or deformation appear after the image was captured.")
        add_section("AI Image Analysis", f"**Model Prediction:** {predicted_class}\n\n**Model Confidence:** {confidence:.2f}%\n\n**Estimated Affected Leaf Area:** {affected:.2f}%\n\n**Estimated Severity:** {severity}\n\nThese values are computer-vision estimates from the uploaded image. They are not laboratory measurements and they do not guarantee the whole field is free of all disease or stress.")
        add_section("Important Note", important_note if important_note else "Healthy classification is limited to the trained model and the uploaded image; it is not a guarantee of absolute crop health.")
        # Do not show the evidence footer in the report UI. The relevant agricultural
        # knowledge is already integrated into the report sections above.
        add_section(
            "Agricultural Knowledge & Evidence",
            "The disease information in this report was integrated from authoritative agricultural references and the local AgriVision knowledge database."
        )
        report = "\n".join(sections)
        wikipedia_section = wikipedia_report_section(predicted_class)
        if wikipedia_section:
            report += "\n" + wikipedia_section.strip() + "\n"
        return report.strip() + "\n"

    # ------------------------------------------------------------------
    # Standard disease report
    # ------------------------------------------------------------------
    add_section("Crop / Plant", crop)
    add_section("Detected Condition", condition)
    add_section("Disease Type", disease_type)
    add_section("Scientific Name / Causal Agent", f"**Causal agent:** {causal_agent}\n\n**Scientific name:** {scientific_name}\n\n**Pathogen group:** {pathogen_group}")
    add_section("Disease Definition", definition)
    add_section("Disease Overview", detailed_description)
    add_section("Why Does This Disease Occur?", why_it_occurs or "The disease develops when susceptible host tissue encounters a viable pathogen and the environment remains favorable for infection.")
    add_section("Disease Triangle", _format_disease_triangle(disease_triangle))
    add_section("Causal Agent Details", f"**Causal agent:** {causal_agent}\n\n**Pathogen group:** {pathogen_group}\n\n**Scientific name:** {scientific_name}")
    add_section("Favorable Environmental Conditions", _format_list(favorable_conditions if favorable_conditions else ["Warm conditions", "High humidity", "Leaf wetness", "Suitable host stage"]))
    add_section("How the Disease Starts", _format_list(how_it_starts if how_it_starts else ["Initial infection begins when a susceptible host tissue is exposed to viable inoculum under favorable moisture and temperature conditions."]))
    add_section("Disease Cycle", disease_cycle or "Source/survival -> initial infection -> pathogen development -> symptom appearance -> spread -> secondary infection -> disease progression.")
    add_section("How the Disease Spreads", _format_list(how_it_spreads if how_it_spreads else ["The pathogen spreads through the crop by wind, splashing water, contaminated tools or handling."]))
    add_section("Early Symptoms", _format_list(early_symptoms if early_symptoms else ["Symptoms can appear first on older or more susceptible tissue."]))
    add_section("Why These Symptoms Occur", _format_symptom_explanations(symptom_explanations if symptom_explanations else [{"symptom": "Visible disease symptoms", "why_it_occurs": "The pathogen damages host tissues and disrupts normal leaf function as infection advances.", "when_it_occurs": "Symptoms appear during periods favorable to disease development.", "what_it_indicates": "The disease is active and tissue function is being reduced."}]))
    add_section("Leaf Symptoms", _format_list(leaf_symptoms if leaf_symptoms else ["Leaf symptoms vary by disease and stage, but they usually reflect active tissue damage and reduced leaf function."]))
    add_section("Stem Symptoms", _format_list(stem_symptoms if stem_symptoms else ["Not a major symptom for this disease."]))
    add_section("Fruit Symptoms", _format_list(fruit_symptoms if fruit_symptoms else ["Not a major symptom for this disease."]))
    add_section("Root / Tuber Symptoms", _format_list(root_or_tuber_symptoms if root_or_tuber_symptoms else ["Not applicable for this disease."]))
    add_section("Disease Progression", disease_progression or "Disease intensity increases as infection cycles continue, reducing plant vigor and the functional leaf area.")
    add_section("Identification", _format_list(identification if identification else ["The condition should be assessed alongside the crop context, disease pressure and visible symptom pattern."]))
    add_section("Similar Conditions", _format_list(similar_conditions if similar_conditions else ["Other disorders can appear similar, so local confirmation is valuable when diagnosis is uncertain."]))
    add_section("Monitoring", _format_list(monitoring if monitoring else ["Inspect plants regularly, especially under weather conditions that favor disease development."]))

    prevention_body = []
    if natural_prevention:
        prevention_body.extend(["**Natural / Cultural Prevention**", _format_list(natural_prevention)])
    if cultural_prevention:
        prevention_body.extend(["\n**Cultural Prevention**", _format_list(cultural_prevention)])
    if not prevention_body:
        prevention_body = ["Use clean planting material, keep the canopy dry where practical, and maintain good field sanitation."]
    add_section("Natural / Cultural Prevention", "\n".join(prevention_body))

    add_section("Management", _format_list(management if management else ["Use locally appropriate crop management and follow agricultural-extension advice. "]))
    add_section("What to Avoid", _format_list(what_to_avoid if what_to_avoid else ["Avoid moving contaminated material, persistent leaf wetness or unmanaged disease pressure."]))
    add_section("What Happens Under Different Weather Conditions", weather_risk or "Weather strongly influences disease intensity by changing leaf wetness, humidity and the speed of pathogen reproduction.")
    add_section("When to Seek Expert Help", when_to_seek_expert_help if when_to_seek_expert_help else "Seek local agricultural or plant-pathology guidance if disease spreads rapidly, the diagnosis is uncertain or crop losses are increasing.")
    add_section("AI Image Analysis", f"**Model Prediction:** {predicted_class}\n\n**Model Confidence:** {confidence:.2f}%\n\n**Estimated Affected Leaf Area:** {affected:.2f}%\n\n**Estimated Severity:** {severity}\n\nThese values are computer-vision estimates from the uploaded image. They are not laboratory measurements and they do not represent a confirmed diagnosis.")
    add_section("Important Note", important_note if important_note else "The AgriVision AI output is an image-based estimate and should be interpreted alongside local crop conditions and expert advice.")
    # Do not show the evidence footer in the report UI. The relevant agricultural
    # knowledge is already integrated into the disease sections above.
    add_section(
        "Agricultural Knowledge & Evidence",
        "The disease information in this report was integrated from authoritative agricultural references and the local AgriVision knowledge database."
    )

    report = "\n".join(sections)

    if detailed_description:
        report = report.replace(
            "**Disease Overview**\n\n" + detailed_description,
            "**Disease Overview**\n\n" + detailed_description,
        )

    wikipedia_section = wikipedia_report_section(predicted_class)
    if wikipedia_section:
        report += "\n" + wikipedia_section.strip() + "\n"

    report += "\n---\n\n### ⚠️ AI Analysis Notice\n\nThis report combines the computer-vision model result with local agricultural knowledge stored in the AgriVision database. The image classification is not a laboratory diagnosis. The estimated affected-leaf percentage is a model-derived visual estimate and should not be interpreted as a field-wide or laboratory measurement. For crop-specific treatment decisions, especially when disease identification is uncertain or the crop is severely affected, consult a qualified agricultural expert or plant pathologist.\n"

    return report.strip() + "\n"


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def generate_farmer_report(image, result):
    """
    Backward-compatible function name.

    The application may still call this function, but no
    external AI service is used anymore.
    """

    return generate_report(result)
