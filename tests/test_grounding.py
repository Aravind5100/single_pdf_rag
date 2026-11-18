from lc_rag.grounding import numbers_in, ungrounded_numbers

CONTEXT = ("[Page 29] Residents will be provided with lock-out assistance for up to two (2) occasions per academic "
           "semester at no charge. Each request after this will result in a $25.00 fee. Quiet hours are 10:00 p.m. "
           "to 8:00 a.m. Alcohol: no more than 1,500 ml of wine. Twelve 12 oz. beers.")


def test_numbers_in_handles_currency_decimals_commas_and_number_words():
    assert numbers_in("a $25.00 fee") == ["25", "00"]
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
