import textstat

def analyze_readability(text):
    # Calculate the Flesch-Kincaid Grade Level
    grade_level = textstat.flesch_kincaid_grade(text)    
    # Calculate the Reading Ease (0-100 scale)
    reading_ease = textstat.flesch_reading_ease(text)
    return grade_level, reading_ease


# Test it out
sample_text = """
"The sun is a massive star made of glowing gas. Because plants need sunlight to grow, the sun is essential for life on Earth. Without its heat, our entire planet would be frozen and dark."
"""
if __name__ == "__main__":
    grade, ease = analyze_readability(sample_text)
    print(f"Flesch-Kincaid Grade Level: {grade:.2f}")
    print(f"Flesch Reading Ease: {ease:.2f}")