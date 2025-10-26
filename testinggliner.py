from gliner import GLiNER

sample_text = (
    "John Doe, a 39 year old software engineer at Acme Solutions, has been working in technology sector for over 15 years. "
    "He currently resides at 123 Maple Street, Springfield Illness and enjoys a balanced lifestyle that includes hiking and "
    "playing guitar in his free time in case of emergencies. His sister, Emily Doe, is his primary contact, reachable at "
    "555-235678. John has always been passionate about coding and has contributed to numerous projects, including a popular "
    "mobile phone app that streamlines project management for small businesses. He is not working. He often collaborates "
    "with his colleagues, such as Jane Smith, the head of the department, to brainstorm innovative solutions for upcoming "
    "software releases."
)

model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
entities = model.predict([sample_text])[0]

print("Detected PII Entities:")
for ent in entities:
    print(f"{ent['label']}: {ent['text']}")
