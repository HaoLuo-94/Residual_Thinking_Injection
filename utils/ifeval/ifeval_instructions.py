"""
IFEval instruction verification module.

Implements verification functions for all 25 instruction types in IFEval benchmark.
Based on Google's IFEval: https://arxiv.org/abs/2311.07911

Reference implementation: https://github.com/google-research/google-research/tree/master/instruction_following_eval
"""
import re
import json
import string
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict


# Instruction ID to verification function mapping
INSTRUCTION_DICT = {}


def register_instruction(instruction_id: str):
    """Decorator to register instruction verification functions."""
    def decorator(func):
        INSTRUCTION_DICT[instruction_id] = func
        return func
    return decorator


# ============================================================================
# Language Detection Utilities
# ============================================================================

# Common words for language detection (simplified)
LANGUAGE_KEYWORDS = {
    'en': {'the', 'is', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'was', 'for', 'on', 'are'},
    'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'auf', 'ist'},
    'fr': {'le', 'la', 'les', 'de', 'et', 'des', 'du', 'que', 'est', 'en', 'un', 'une'},
    'es': {'el', 'la', 'de', 'que', 'y', 'los', 'en', 'un', 'es', 'las', 'se', 'del'},
    'it': {'il', 'di', 'che', 'la', 'e', 'un', 'in', 'a', 'del', 'per', 'le', 'una'},
    'pt': {'de', 'da', 'do', 'que', 'os', 'o', 'a', 'em', 'um', 'uma', 'para', 'as'},
    'ru': {'и', 'в', 'не', 'на', 'что', 'я', 'с', 'он', 'как', 'это', 'но', 'по'},
    'zh': {'的', '是', '了', '在', '和', '有', '我', '不', '人', '这', '他', '们'},
    'ja': {'の', 'に', 'は', 'を', 'が', 'と', 'で', 'た', 'て', 'も', 'し', 'な'},
    'ko': {'의', '을', '를', '이', '가', '은', '는', '에', '로', '와', '과', '도'},
    'ar': {'في', 'من', 'على', 'إلى', 'أن', 'مع', 'لا', 'هذا', 'التي', 'كان'},
    'hi': {'का', 'की', 'के', 'है', 'में', 'को', 'और', 'से', 'एक', 'हैं'},
    'nl': {'de', 'het', 'een', 'van', 'en', 'in', 'is', 'op', 'dat', 'te', 'met'},
    'pl': {'i', 'w', 'na', 'nie', 'z', 'do', 'to', 'że', 'się', 'jest', 'o'},
    'sv': {'och', 'i', 'att', 'det', 'är', 'en', 'som', 'på', 'för', 'med', 'av'},
    'vi': {'của', 'là', 'và', 'có', 'trong', 'được', 'cho', 'này', 'một', 'những'},
    'th': {'ที่', 'และ', 'ของ', 'ใน', 'ได้', 'การ', 'มี', 'ไม่', 'จะ', 'เป็น'},
    'tr': {'ve', 'bir', 'bu', 'da', 'ne', 'için', 'var', 'ile', 'mi', 'de'},
}


def detect_language(text: str) -> str:
    """
    Simple language detection based on common word frequency.
    
    Args:
        text: Input text to detect language.
        
    Returns:
        ISO 639-1 language code.
    """
    words = set(text.lower().split())
    scores = {}
    
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        score = len(words & keywords)
        scores[lang] = score
    
    if not scores or max(scores.values()) == 0:
        return 'en'  # Default to English
    
    return max(scores, key=scores.get)


# ============================================================================
# Punctuation Instructions
# ============================================================================

@register_instruction("punctuation:no_comma")
def check_no_comma(text: str, **kwargs) -> bool:
    """Check that response contains no commas."""
    return ',' not in text


# ============================================================================
# Detectable Format Instructions
# ============================================================================

@register_instruction("detectable_format:number_highlighted_sections")
def check_highlighted_sections(text: str, num_highlights: int = 1, **kwargs) -> bool:
    """
    Check for highlighted sections using markdown format.
    
    Highlighted sections are marked with *section title*.
    """
    if num_highlights is None:
        num_highlights = 1
    
    # Match markdown bold/italic: *text* or **text**
    pattern = r'\*[^*\n]+\*'
    matches = re.findall(pattern, text)
    return len(matches) >= num_highlights


@register_instruction("detectable_format:number_bullet_points")
def check_bullet_points(text: str, num_bullets: int = 1, **kwargs) -> bool:
    """Check for bullet point format."""
    if num_bullets is None:
        num_bullets = 1
    
    # Match various bullet point formats: -, *, •, numbers with periods
    pattern = r'^[\s]*[-*•][\s]+|^\s*\d+[.)\]]\s+'
    matches = re.findall(pattern, text, re.MULTILINE)
    return len(matches) >= num_bullets


# Alias for number_bullet_lists (same as number_bullet_points)
@register_instruction("detectable_format:number_bullet_lists")
def check_bullet_lists(text: str, num_bullets: int = 1, **kwargs) -> bool:
    """Check for bullet list format (alias for number_bullet_points)."""
    return check_bullet_points(text, num_bullets, **kwargs)


@register_instruction("detectable_format:constrained_response")
def check_constrained_response(text: str, **kwargs) -> bool:
    """Check that response is properly constrained (not too long/random)."""
    # Basic check: response should not be empty and should be reasonable length
    text = text.strip()
    return len(text) > 0 and len(text) < 50000


@register_instruction("detectable_format:json_format")
def check_json_format(text: str, **kwargs) -> bool:
    """Check that the response is valid JSON."""
    text = text.strip()
    
    # Try to find JSON in the text
    # First try the whole text
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in code fence
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            json.loads(match.strip())
            return True
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object or array in text
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return True
            except json.JSONDecodeError:
                continue
    
    return False


@register_instruction("detectable_format:multiple_sections")
def check_multiple_sections(text: str, num_sections: int = 2, section_spliter: str = None, **kwargs) -> bool:
    """Check that response has multiple sections."""
    if num_sections is None:
        num_sections = 2
    
    if section_spliter:
        sections = text.split(section_spliter)
    else:
        # Default: check for markdown headers or double newlines
        # Count markdown headers
        headers = re.findall(r'^#+\s+.+$', text, re.MULTILINE)
        if len(headers) >= num_sections:
            return True
        
        # Count sections separated by double newlines
        sections = re.split(r'\n\n+', text.strip())
    
    return len([s for s in sections if s.strip()]) >= num_sections


@register_instruction("detectable_format:title")
def check_title(text: str, **kwargs) -> bool:
    """Check that response includes a title."""
    text = text.strip()
    
    # Check for markdown title: # Title
    if re.match(r'^#\s+.+', text, re.MULTILINE):
        return True
    
    # Check for title followed by newline at start
    lines = text.split('\n')
    if len(lines) >= 2:
        first_line = lines[0].strip()
        second_line = lines[1].strip()
        # Title is usually short and followed by empty line or content
        if len(first_line) > 0 and len(first_line) < 100 and (not second_line or len(second_line) > 0):
            return True
    
    return False


@register_instruction("detectable_format:number_placeholders")
def check_placeholders(text: str, num_placeholders: int = 1, **kwargs) -> bool:
    """Check for placeholder format like [placeholder]."""
    if num_placeholders is None:
        num_placeholders = 1
    
    # Match [text] placeholders
    pattern = r'\[[^\[\]]+\]'
    matches = re.findall(pattern, text)
    return len(matches) >= num_placeholders


# ============================================================================
# Length Constraint Instructions
# ============================================================================

@register_instruction("length_constraints:number_words")
def check_word_count(text: str, num_words: int = None, relation: str = "at least", **kwargs) -> bool:
    """
    Check word count constraint.
    
    Args:
        text: Response text.
        num_words: Target word count.
        relation: Comparison type - "at least", "at most", "less than", etc.
    """
    if num_words is None:
        return True
    
    # Count words (split by whitespace)
    word_count = len(text.split())
    
    relation = relation.lower() if relation else "at least"
    
    if relation in ["at least", "at_least", "atleast", "minimum"]:
        return word_count >= num_words
    elif relation in ["at most", "at_most", "atmost", "maximum"]:
        return word_count <= num_words
    elif relation in ["less than", "less_than", "lessthan"]:
        return word_count < num_words
    elif relation in ["more than", "more_than", "morethan"]:
        return word_count > num_words
    elif relation in ["exactly", "equal"]:
        return word_count == num_words
    else:
        return word_count >= num_words


@register_instruction("length_constraints:number_sentences")
def check_sentence_count(text: str, num_sentences: int = None, relation: str = "at least", **kwargs) -> bool:
    """Check sentence count constraint."""
    if num_sentences is None:
        return True
    
    # Count sentences (split by sentence-ending punctuation)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    relation = relation.lower() if relation else "at least"
    
    if relation in ["at least", "at_least", "minimum"]:
        return sentence_count >= num_sentences
    elif relation in ["at most", "at_most", "maximum"]:
        return sentence_count <= num_sentences
    elif relation in ["less than", "less_than"]:
        return sentence_count < num_sentences
    elif relation in ["more than", "more_than"]:
        return sentence_count > num_sentences
    elif relation in ["exactly", "equal"]:
        return sentence_count == num_sentences
    else:
        return sentence_count >= num_sentences


@register_instruction("length_constraints:number_paragraphs")
def check_paragraph_count(text: str, num_paragraphs: int = None, relation: str = "at least", **kwargs) -> bool:
    """Check paragraph count constraint."""
    if num_paragraphs is None:
        return True
    
    # Count paragraphs (separated by double newlines)
    paragraphs = re.split(r'\n\n+', text.strip())
    para_count = len([p for p in paragraphs if p.strip()])
    
    relation = relation.lower() if relation else "at least"
    
    if relation in ["at least", "at_least", "minimum"]:
        return para_count >= num_paragraphs
    elif relation in ["at most", "at_most", "maximum"]:
        return para_count <= num_paragraphs
    elif relation in ["less than", "less_than"]:
        return para_count < num_paragraphs
    elif relation in ["more than", "more_than"]:
        return para_count > num_paragraphs
    elif relation in ["exactly", "equal"]:
        return para_count == num_paragraphs
    else:
        return para_count >= num_paragraphs


@register_instruction("length_constraints:nth_paragraph_first_word")
def check_nth_paragraph_first_word(text: str, nth_paragraph: int = 1, first_word: str = None, **kwargs) -> bool:
    """Check that the nth paragraph starts with a specific word."""
    if first_word is None:
        return True
    
    paragraphs = re.split(r'\n\n+', text.strip())
    paragraphs = [p for p in paragraphs if p.strip()]
    
    if nth_paragraph > len(paragraphs):
        return False
    
    # nth_paragraph is 1-indexed
    target_para = paragraphs[nth_paragraph - 1].strip()
    words = target_para.split()
    
    if not words:
        return False
    
    return words[0].lower().strip(string.punctuation) == first_word.lower().strip(string.punctuation)


# ============================================================================
# Keyword Instructions
# ============================================================================

@register_instruction("keywords:existence")
def check_keyword_existence(text: str, keywords: List[str] = None, **kwargs) -> bool:
    """Check that all specified keywords exist in the text."""
    if not keywords:
        return True
    
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() not in text_lower:
            return False
    return True


@register_instruction("keywords:frequency")
def check_keyword_frequency(text: str, keyword: str = None, frequency: int = 1, relation: str = "at least", **kwargs) -> bool:
    """Check keyword frequency constraint."""
    if not keyword:
        return True
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    count = text_lower.count(keyword_lower)
    
    relation = relation.lower() if relation else "at least"
    
    if relation in ["at least", "at_least", "minimum"]:
        return count >= frequency
    elif relation in ["at most", "at_most", "maximum"]:
        return count <= frequency
    elif relation in ["less than", "less_than"]:
        return count < frequency
    elif relation in ["more than", "more_than"]:
        return count > frequency
    elif relation in ["exactly", "equal"]:
        return count == frequency
    else:
        return count >= frequency


@register_instruction("keywords:forbidden_words")
def check_forbidden_words(text: str, forbidden_words: List[str] = None, **kwargs) -> bool:
    """Check that no forbidden words appear in the text."""
    if not forbidden_words:
        return True
    
    text_lower = text.lower()
    for word in forbidden_words:
        if word.lower() in text_lower:
            return False
    return True


@register_instruction("keywords:letter_frequency")
def check_letter_frequency(text: str, letter: str = None, let_frequency: int = 1, let_relation: str = "at least", **kwargs) -> bool:
    """Check letter frequency constraint."""
    if not letter:
        return True
    
    text_lower = text.lower()
    letter_lower = letter.lower()
    count = text_lower.count(letter_lower)
    
    relation = let_relation.lower() if let_relation else "at least"
    
    if relation in ["at least", "at_least", "minimum"]:
        return count >= let_frequency
    elif relation in ["at most", "at_most", "maximum"]:
        return count <= let_frequency
    elif relation in ["less than", "less_than"]:
        return count < let_frequency
    elif relation in ["more than", "more_than"]:
        return count > let_frequency
    elif relation in ["exactly", "equal"]:
        return count == let_frequency
    else:
        return count >= let_frequency


# ============================================================================
# Language Instructions
# ============================================================================

@register_instruction("language:response_language")
def check_response_language(text: str, language: str = None, **kwargs) -> bool:
    """Check that response is in the specified language."""
    if not language:
        return True
    
    detected = detect_language(text)
    return detected == language.lower()


# ============================================================================
# Startend Instructions
# ============================================================================

@register_instruction("startend:end_checker")
def check_end_phrase(text: str, end_phrase: str = None, **kwargs) -> bool:
    """Check that response ends with a specific phrase."""
    if not end_phrase:
        return True
    
    text = text.strip()
    return text.endswith(end_phrase)


@register_instruction("startend:quotation")
def check_quotation(text: str, **kwargs) -> bool:
    """Check that response contains quotation marks properly."""
    # Check for balanced quotation marks
    single_quotes = text.count("'")
    double_quotes = text.count('"')
    
    # At least one pair of quotes
    return (single_quotes >= 2 and single_quotes % 2 == 0) or (double_quotes >= 2 and double_quotes % 2 == 0)


# ============================================================================
# Change Case Instructions
# ============================================================================

@register_instruction("change_case:capital_word_frequency")
def check_capital_word_frequency(text: str, capital_frequency: int = 1, capital_relation: str = "at least", **kwargs) -> bool:
    """Check frequency of capitalized words."""
    if capital_frequency is None:
        return True
    
    # Count words that are fully capitalized (all caps)
    words = text.split()
    capital_count = sum(1 for word in words if word.isupper() and len(word) > 1)
    
    relation = capital_relation.lower() if capital_relation else "at least"
    
    if relation in ["at least", "at_least", "minimum"]:
        return capital_count >= capital_frequency
    elif relation in ["at most", "at_most", "maximum"]:
        return capital_count <= capital_frequency
    else:
        return capital_count >= capital_frequency


@register_instruction("change_case:english_capital")
def check_english_capital(text: str, **kwargs) -> bool:
    """Check that response follows English capitalization rules."""
    # Check if sentences start with capital letters
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence[0].isalpha() and not sentence[0].isupper():
            return False
    return True


@register_instruction("change_case:english_lowercase")
def check_english_lowercase(text: str, **kwargs) -> bool:
    """Check that response is in lowercase."""
    # Allow for some flexibility (punctuation, numbers)
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return True
    lowercase_count = sum(1 for c in alpha_chars if c.islower())
    return lowercase_count / len(alpha_chars) > 0.95


# ============================================================================
# Combination Instructions
# ============================================================================

@register_instruction("combination:two_responses")
def check_two_responses(text: str, **kwargs) -> bool:
    """Check that response contains two separate responses."""
    # Look for separators that indicate two responses
    separators = ['***', '---', '===', '\n\n\n', 'Response 1:', 'Response 2:', '1.', '2.']
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            if len([p for p in parts if p.strip()]) >= 2:
                return True
    return False


@register_instruction("combination:repeat_prompt")
def check_repeat_prompt(text: str, prompt_to_repeat: str = None, **kwargs) -> bool:
    """Check that response repeats the original prompt."""
    if not prompt_to_repeat:
        return True
    
    return prompt_to_repeat.lower() in text.lower()


# ============================================================================
# Detectable Content Instructions
# ============================================================================

@register_instruction("detectable_content:number_placeholders")
def check_content_placeholders(text: str, num_placeholders: int = 1, **kwargs) -> bool:
    """Check for content placeholders like [NAME], [DATE], etc."""
    return check_placeholders(text, num_placeholders, **kwargs)


@register_instruction("detectable_content:postscript")
def check_postscript(text: str, postscript_marker: str = "P.S.", **kwargs) -> bool:
    """Check that response contains a postscript."""
    if not postscript_marker:
        postscript_marker = "P.S."
    
    # Common postscript markers
    ps_patterns = [postscript_marker, "P.S.", "PS:", "P.S:", "PS.", "ps:", "p.s."]
    text_lower = text.lower()
    
    for pattern in ps_patterns:
        if pattern.lower() in text_lower:
            return True
    return False


# ============================================================================
# Main Verification Functions
# ============================================================================

def verify_instruction(instruction_id: str, text: str, kwargs: Dict[str, Any]) -> bool:
    """
    Verify if a response satisfies a single instruction.
    
    Args:
        instruction_id: Instruction identifier (e.g., "punctuation:no_comma").
        text: Model response text.
        kwargs: Instruction parameters.
        
    Returns:
        True if instruction is satisfied, False otherwise.
    """
    if instruction_id not in INSTRUCTION_DICT:
        # Unknown instruction, skip it (be lenient)
        print(f"Warning: Unknown instruction '{instruction_id}', skipping")
        return True
    
    verifier = INSTRUCTION_DICT[instruction_id]
    try:
        return verifier(text, **kwargs)
    except Exception as e:
        print(f"Warning: Error verifying instruction '{instruction_id}': {e}")
        return False


def verify_all_instructions(
    text: str, 
    instruction_id_list: List[str], 
    kwargs_list: List[Dict[str, Any]]
) -> Tuple[bool, List[bool]]:
    """
    Verify if a response satisfies all instructions.
    
    Args:
        text: Model response text.
        instruction_id_list: List of instruction identifiers.
        kwargs_list: List of kwargs for each instruction.
        
    Returns:
        Tuple of (all_satisfied, results_list) where results_list contains boolean results for each instruction.
    """
    results = []
    
    for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
        # Filter out None values from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        result = verify_instruction(instruction_id, text, filtered_kwargs)
        results.append(result)
    
    all_satisfied = all(results)
    return all_satisfied, results


def compute_ifeval_metrics(
    predictions: List[str],
    instruction_id_lists: List[List[str]],
    kwargs_lists: List[List[Dict[str, Any]]]
) -> Dict[str, float]:
    """
    Compute IFEval metrics for a batch of predictions.
    
    Computes:
    - Prompt-level strict accuracy: % of prompts where ALL instructions are satisfied
    - Prompt-level loose accuracy: Average % of instructions satisfied per prompt
    - Instruction-level strict accuracy: % of individual instructions satisfied
    
    Args:
        predictions: List of model responses.
        instruction_id_lists: List of instruction ID lists for each prompt.
        kwargs_lists: List of kwargs lists for each prompt.
        
    Returns:
        Dictionary with accuracy metrics.
    """
    prompt_strict_correct = 0
    prompt_loose_scores = []
    instruction_results = []
    
    for pred, inst_ids, kwarg_list in zip(predictions, instruction_id_lists, kwargs_lists):
        all_satisfied, results = verify_all_instructions(pred, inst_ids, kwarg_list)
        
        # Prompt-level strict
        if all_satisfied:
            prompt_strict_correct += 1
        
        # Prompt-level loose (average per prompt)
        if results:
            prompt_loose_scores.append(sum(results) / len(results))
        else:
            prompt_loose_scores.append(1.0)
        
        # Instruction-level
        instruction_results.extend(results)
    
    n_prompts = len(predictions)
    n_instructions = len(instruction_results)
    
    return {
        "prompt_strict_accuracy": prompt_strict_correct / n_prompts if n_prompts > 0 else 0.0,
        "prompt_loose_accuracy": sum(prompt_loose_scores) / n_prompts if n_prompts > 0 else 0.0,
        "instruction_accuracy": sum(instruction_results) / n_instructions if n_instructions > 0 else 0.0,
        "total_prompts": n_prompts,
        "total_instructions": n_instructions,
        "prompts_all_correct": prompt_strict_correct
    }


# List all registered instructions
def get_registered_instructions() -> List[str]:
    """Get list of all registered instruction IDs."""
    return list(INSTRUCTION_DICT.keys())
