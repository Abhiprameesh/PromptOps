from app.llm.parser import parse_llm_response

response = """
```json
{
  "category": "bug",
  "priority": "high",
  "summary": "Users are unable to log in after a successful password reset."
}

"""

result = parse_llm_response(response)

print("=" * 50)
print("Parsed Response")
print("=" * 50)
print(result.model_dump_json(indent=4))