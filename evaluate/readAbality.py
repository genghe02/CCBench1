from textstat import textstat

test_data = "Playing games has always been thought to be important to "


l = [textstat.flesch_reading_ease(test_data),
textstat.flesch_kincaid_grade(test_data),
textstat.smog_index(test_data),
textstat.coleman_liau_index(test_data),
textstat.automated_readability_index(test_data),
textstat.dale_chall_readability_score(test_data),
textstat.difficult_words(test_data),
textstat.linsear_write_formula(test_data),
textstat.gunning_fog(test_data)]

for i in l:
    print(i)