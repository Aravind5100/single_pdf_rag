from lc_rag.grounding import numbers_in, ungrounded_numbers

CONTEXT = ("[Page 29] Residents will be provided with lock-out assistance for up to two (2) occasions per academic "
           "semester at no charge. Each request after this will result in a $25.00 fee. Quiet hours are 10:00 p.m. "
           "to 8:00 a.m. Alcohol: no more than 1,500 ml of wine. Twelve 12 oz. beers.")


def test_numbers_in_handles_currency_decimals_commas_and_number_words():
    assert numbers_in("a $25.00 fee") == ["25"]                # trailing ".00" is the same amount
    assert numbers_in("1,500 ml") == ["1500"]
    assert numbers_in("limited to five visits and twelve beers") == ["5", "12"]
    assert numbers_in("one of the following") == []            # 'one' is not treated as a quantity


def test_answer_side_ignores_citations_and_list_markers():
    answer = "1. First step [Page 15]\n2. Second step (Page 16), see page 17."
    assert numbers_in(answer, answer_side=True) == []
    assert numbers_in(answer) != []                              # the context side keeps every digit


def test_supported_numbers_pass():
    answer = "Residents get two lock-outs per semester, then a $25.00 fee. [Page 29]"
    assert ungrounded_numbers(answer, CONTEXT) == []


def test_invented_numbers_are_reported_once_in_order():
    answer = "A $40 fee applies, plus a $100 fine, and the $40 is annual."
    assert ungrounded_numbers(answer, CONTEXT) == ["40", "100"]


def test_number_words_in_the_answer_are_checked_too():
    assert ungrounded_numbers("Each student is limited to exactly five visits.", CONTEXT) == ["5"]
    assert ungrounded_numbers("There are twelve beers allowed", CONTEXT) == []


def test_time_written_without_minutes_matches_the_context_time():
    assert ungrounded_numbers("Quiet hours start at 10 p.m. and end at 8 a.m.", CONTEXT) == []


def test_a_page_citation_for_an_unrelated_page_is_not_a_claim():
    assert ungrounded_numbers("No adjustment is made. [Page 99]", CONTEXT) == []


def test_it_cannot_catch_an_invented_fact_without_a_number():
    # Documented limitation: no digits, so nothing to compare.
    assert ungrounded_numbers("The final decision is made by a student panel vote.", CONTEXT) == []


# ---- false-rejection classes found in review: each would wrongly refuse a correct answer ----

def test_compound_number_words_become_one_number():
    assert numbers_in("Twenty-four-hour quiet hours") == ["24"]
    assert numbers_in("twenty one years and thirty five days") == ["21", "35"]
    assert numbers_in("fifteen and sixty") == ["15", "60"]        # a tens word alone is a number too
    assert numbers_in("sixty-five") == ["65"]


def test_the_handbooks_own_spelling_of_24_hour_matches_a_digit_answer_and_back():
    assert ungrounded_numbers("24-hour quiet hours apply", "Twenty-four-hour quiet hours begin on Friday") == []
    assert ungrounded_numbers("Twenty-four-hour quiet hours apply", "24-hour quiet hours begin on Friday") == []
    assert ungrounded_numbers("Twenty-one is the age", "You must be 21 or older") == []


def test_a_number_echoed_from_the_question_counts_as_grounded():
    assert ungrounded_numbers("No, residents under 18 are not covered.", "Residents must follow policy.",
                              question="Can residents under 18 stay?") == []
    assert ungrounded_numbers("A $50 fine applies.", "Residents must follow policy.",
                              question="Can residents under 18 stay?") == ["50"]


def test_leading_zeros_and_trailing_decimals_do_not_cause_false_flags():
    assert ungrounded_numbers("Check-in is at 08:00.", "Check-in is at 8:00 a.m.") == []
    assert ungrounded_numbers("The fee is $25.00.", "The fee is $25.") == []
    assert ungrounded_numbers("The fee is $25.", "The fee is $25.00.") == []
    assert ungrounded_numbers("The fee is $25.50.", "The fee is $25.") == ["50"]      # a different amount still flags


def test_markdown_list_markers_and_inline_enumerators_are_formatting():
    answer = "**1.** Submit the form.\n- 2. Wait.\n3) Appeal (1) in writing and (2) on time."
    assert numbers_in(answer, answer_side=True) == []


def test_page_citation_variants_are_ignored():
    for citation in ("[Pages 18 and 25]", "[Page 19\u201320]", "p. 19", "pp. 18-19", "(page 7)"):
        assert numbers_in(f"No fee applies. {citation}", answer_side=True) == [], citation


def test_a_retrieved_page_label_number_is_not_grounding():
    # Callers pass chunk text, not text with '[Page 19]' labels added, so '19' here is genuinely ungrounded.
    assert ungrounded_numbers("A $19 fee applies.", "Residents must follow the policy.") == ["19"]
