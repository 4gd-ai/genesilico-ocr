from typing import Any

def normalize_array_fields(trf_data: dict) -> dict:
    def is_numeric_dict(d: Any) -> bool:
        return isinstance(d, dict) and all(str(k).isdigit() for k in d.keys())

    array_fields = [
        "Sample",
        "FamilyHistory.familyMember"
    ]

    for field in array_fields:
        parts = field.split(".")
        obj = trf_data
        for part in parts[:-1]:
            obj = obj.get(part, {})
        last_key = parts[-1]

        if isinstance(obj, dict) and last_key in obj:
            value = obj[last_key]
            if is_numeric_dict(value):
                obj[last_key] = [v for _, v in sorted(value.items(), key=lambda x: int(x[0]))]
            elif isinstance(value, str):
                obj[last_key] = [value.strip()] if value.strip() else []

    # ✅ Ensure Sample is always a list of dicts if mistakenly returned as dict
    sample_data = trf_data.get("Sample")
    if isinstance(sample_data, dict):
        trf_data["Sample"] = [sample_data]

    return trf_data
