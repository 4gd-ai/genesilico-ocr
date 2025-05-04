def normalize_array_fields(trf_data: dict) -> dict:
    def is_numeric_dict(d: dict) -> bool:
        return isinstance(d, dict) and all(k.isdigit() for k in d.keys())

    array_fields = ["Sample", "FamilyHistory.familyMember"]

    for field in array_fields:
        parts = field.split(".")
        obj = trf_data
        for part in parts[:-1]:
            obj = obj.get(part, {})
        last_key = parts[-1]

        if last_key in obj and is_numeric_dict(obj[last_key]):
            obj[last_key] = [v for _, v in sorted(obj[last_key].items(), key=lambda x: int(x[0]))]

    return trf_data