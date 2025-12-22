def snakecase_to_pascalcase(s: str) -> str:
    components = s.split("_")
    return "".join(word.capitalize() for word in components)
