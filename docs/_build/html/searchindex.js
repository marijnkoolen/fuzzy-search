Search.setIndex({docnames:["index","source/fuzzy_search","source/modules","source/setup","source/test"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,sphinx:56},filenames:["index.rst","source/fuzzy_search.rst","source/modules.rst","source/setup.rst","source/test.rst"],objects:{"":{fuzzy_search:[1,0,0,"-"],test:[4,0,0,"-"]},"fuzzy_search.fuzzy_context_searcher":{FuzzyContextSearcher:[1,1,1,""]},"fuzzy_search.fuzzy_context_searcher.FuzzyContextSearcher":{add_match_context:[1,2,1,""],configure_context:[1,2,1,""],find_matches:[1,2,1,""],find_matches_in_context:[1,2,1,""]},"fuzzy_search.fuzzy_match":{Candidate:[1,1,1,""],PhraseMatch:[1,1,1,""],PhraseMatchInContext:[1,1,1,""],adjust_match_end_offset:[1,3,1,""],adjust_match_offsets:[1,3,1,""],adjust_match_start_offset:[1,3,1,""],calculate_end_shift:[1,3,1,""],map_string:[1,3,1,""],validate_match_props:[1,3,1,""]},"fuzzy_search.fuzzy_match.Candidate":{add_skip_match:[1,2,1,""],get_match_start_offset:[1,2,1,""],get_match_string:[1,2,1,""],get_skip_count_overlap:[1,2,1,""],get_skip_set_overlap:[1,2,1,""],is_match:[1,2,1,""],remove_first_skip:[1,2,1,""],same_candidate:[1,2,1,""],shift_start_skip:[1,2,1,""],skip_match_length:[1,2,1,""]},"fuzzy_search.fuzzy_match.PhraseMatch":{add_scores:[1,2,1,""],as_web_anno:[1,2,1,""],json:[1,2,1,""],overlaps:[1,2,1,""],score_character_overlap:[1,2,1,""],score_levenshtein_similarity:[1,2,1,""],score_ngram_overlap:[1,2,1,""]},"fuzzy_search.fuzzy_match.PhraseMatchInContext":{json:[1,2,1,""]},"fuzzy_search.fuzzy_patterns":{context_before_pattern:[1,3,1,""],context_then_pattern:[1,3,1,""],escape_string:[1,3,1,""],get_context_patterns:[1,3,1,""],get_search_patterns:[1,3,1,""],list_context_pattern_types:[1,3,1,""],list_pattern_definitions:[1,3,1,""],list_pattern_names:[1,3,1,""],make_search_context_patterns:[1,3,1,""],pattern_before_context:[1,3,1,""],pattern_comma_then_context:[1,3,1,""]},"fuzzy_search.fuzzy_phrase":{Phrase:[1,1,1,""],is_valid_label:[1,3,1,""]},"fuzzy_search.fuzzy_phrase.Phrase":{add_max_offset:[1,2,1,""],add_metadata:[1,2,1,""],has_label:[1,2,1,""],has_skipgram:[1,2,1,""],is_early_skipgram:[1,2,1,""],set_label:[1,2,1,""],skipgram_offsets:[1,2,1,""],within_range:[1,2,1,""]},"fuzzy_search.fuzzy_phrase_model":{PhraseModel:[1,1,1,""],as_phrase_object:[1,3,1,""],is_phrase_dict:[1,3,1,""]},"fuzzy_search.fuzzy_phrase_model.PhraseModel":{add_custom:[1,2,1,""],add_distractor:[1,2,1,""],add_distractors:[1,2,1,""],add_labels:[1,2,1,""],add_model:[1,2,1,""],add_phrase:[1,2,1,""],add_phrases:[1,2,1,""],add_variant:[1,2,1,""],add_variants:[1,2,1,""],get:[1,2,1,""],get_labels:[1,2,1,""],get_phrases:[1,2,1,""],get_phrases_by_max_length:[1,2,1,""],get_variants:[1,2,1,""],has_custom:[1,2,1,""],has_label:[1,2,1,""],has_phrase:[1,2,1,""],index_phrase_words:[1,2,1,""],is_label:[1,2,1,""],json:[1,2,1,""],remove_custom:[1,2,1,""],remove_distractor:[1,2,1,""],remove_distractors:[1,2,1,""],remove_labels:[1,2,1,""],remove_phrase:[1,2,1,""],remove_phrase_words:[1,2,1,""],remove_phrases:[1,2,1,""],remove_variant:[1,2,1,""],remove_variants:[1,2,1,""],validate_entry_phrase:[1,2,1,""],variant_of:[1,2,1,""],variants:[1,2,1,""]},"fuzzy_search.fuzzy_phrase_searcher":{FuzzyPhraseSearcher:[1,1,1,""],SkipMatches:[1,1,1,""],candidates_to_matches:[1,3,1,""],filter_overlapping_phrase_candidates:[1,3,1,""],filter_skipgram_threshold:[1,3,1,""],get_exact_match_ranges:[1,3,1,""],get_known_word_offsets:[1,3,1,""],get_skipmatch_candidates:[1,3,1,""],get_skipmatch_phrase_candidates:[1,3,1,""],get_skipset_overlap:[1,3,1,""],get_text_dict:[1,3,1,""],index_known_word_offsets:[1,3,1,""],search_exact:[1,3,1,""],search_exact_phrases:[1,3,1,""],search_exact_phrases_with_word_boundaries:[1,3,1,""],search_exact_phrases_without_word_boundaries:[1,3,1,""]},"fuzzy_search.fuzzy_phrase_searcher.FuzzyPhraseSearcher":{configure:[1,2,1,""],filter_matches_by_distractors:[1,2,1,""],filter_matches_by_threshold:[1,2,1,""],find_candidates:[1,2,1,""],find_exact_matches:[1,2,1,""],find_matches:[1,2,1,""],find_skipgram_matches:[1,2,1,""],index_distractors:[1,2,1,""],index_phrase_model:[1,2,1,""],index_phrases:[1,2,1,""],index_variants:[1,2,1,""],set_strip_suffix:[1,2,1,""]},"fuzzy_search.fuzzy_phrase_searcher.SkipMatches":{add_skip_match:[1,2,1,""]},"fuzzy_search.fuzzy_searcher":{FuzzySearcher:[1,1,1,""],create_term_match:[1,3,1,""]},"fuzzy_search.fuzzy_searcher.FuzzySearcher":{disable_strip_suffix:[1,2,1,""],enable_strip_suffix:[1,2,1,""],filter_candidates:[1,2,1,""],filter_char_match_candidates:[1,2,1,""],filter_levenshtein_candidates:[1,2,1,""],filter_ngram_candidates:[1,2,1,""],find_candidates:[1,2,1,""],find_start_candidates:[1,2,1,""],find_term_matches:[1,2,1,""],make_ngrams:[1,2,1,""],rank_candidates:[1,2,1,""],score_char_overlap:[1,2,1,""],score_char_overlap_ratio:[1,2,1,""],score_levenshtein_distance:[1,2,1,""],score_levenshtein_distance_ratio:[1,2,1,""],score_ngram_overlap:[1,2,1,""],score_ngram_overlap_ratio:[1,2,1,""],strip_suffix:[1,2,1,""]},"fuzzy_search.fuzzy_string":{SkipGram:[1,1,1,""],get_non_word_prefix:[1,3,1,""],get_non_word_suffix:[1,3,1,""],insert_skips:[1,3,1,""],make_ngrams:[1,3,1,""],score_char_overlap:[1,3,1,""],score_char_overlap_ratio:[1,3,1,""],score_levenshtein_distance:[1,3,1,""],score_levenshtein_similarity_ratio:[1,3,1,""],score_ngram_overlap:[1,3,1,""],score_ngram_overlap_ratio:[1,3,1,""],strip_prefix:[1,3,1,""],strip_suffix:[1,3,1,""],text2skipgrams:[1,3,1,""]},"fuzzy_search.fuzzy_template":{FuzzyTemplate:[1,1,1,""],FuzzyTemplateElement:[1,1,1,""],FuzzyTemplateGroupElement:[1,1,1,""],FuzzyTemplateLabelElement:[1,1,1,""],generate_group_from_json:[1,3,1,""],generate_label_from_json:[1,3,1,""],validate_element_properties:[1,3,1,""]},"fuzzy_search.fuzzy_template.FuzzyTemplate":{get_element:[1,2,1,""],get_elements_by_cardinality:[1,2,1,""],get_label_phrases:[1,2,1,""],get_labels_by_cardinality:[1,2,1,""],get_required_elements:[1,2,1,""],get_required_labels:[1,2,1,""],has_group:[1,2,1,""],has_label:[1,2,1,""],parse_group_element:[1,2,1,""],parse_label_element:[1,2,1,""],register_template:[1,2,1,""]},"fuzzy_search.fuzzy_template_searcher":{FuzzyTemplateSearcher:[1,1,1,""],TemplateMatch:[1,1,1,""],find_next_element_end_index:[1,3,1,""],find_next_element_start_index:[1,3,1,""],find_next_group_match_sequence:[1,3,1,""],find_next_ordered_group_match_sequence:[1,3,1,""],find_next_unordered_group_match_sequence:[1,3,1,""],get_phrase_match_list_labels:[1,3,1,""],get_sequence_label_element_matches:[1,3,1,""],has_required_matches:[1,3,1,""],initialize_sequence:[1,3,1,""],share_label:[1,3,1,""]},"fuzzy_search.fuzzy_template_searcher.FuzzyTemplateSearcher":{filter_phrase_matches:[1,2,1,""],find_template_matches:[1,2,1,""],search_text:[1,2,1,""],set_template:[1,2,1,""]},"test.test_fuzzy_context_searcher":{TestFuzzyContextSearcher:[4,1,1,""]},"test.test_fuzzy_context_searcher.TestFuzzyContextSearcher":{setUp:[4,2,1,""],test_fuzzy_context_searcher:[4,2,1,""],test_fuzzy_context_searcher_can_add_context:[4,2,1,""],test_fuzzy_context_searcher_can_search_context:[4,2,1,""],test_fuzzy_context_searcher_can_set_context_size:[4,2,1,""],test_fuzzy_context_searcher_finds_match_with_context:[4,2,1,""]},"test.test_fuzzy_match":{TestFuzzyMatch:[4,1,1,""],TestMatchInContext:[4,1,1,""]},"test.test_fuzzy_match.TestFuzzyMatch":{test_adjust_boundaries_finds_word_boundary:[4,2,1,""],test_adjust_boundaries_removes_surrounding_whitespace:[4,2,1,""],test_adjust_end:[4,2,1,""],test_adjust_end_does_not_shift_when_end_middle_of_next_word:[4,2,1,""],test_adjust_end_shifts_back_one_when_ending_with_whitespace:[4,2,1,""],test_adjust_end_shifts_back_one_when_phrase_ends_with_whitespace:[4,2,1,""],test_adjust_end_shifts_back_two_when_ending_with_whitespace_and_char:[4,2,1,""],test_adjust_end_shifts_to_end_of_next_word:[4,2,1,""],test_adjust_start:[4,2,1,""],test_adjust_start_does_not_shift_to_third_previous_character:[4,2,1,""],test_adjust_start_returns_when_in_middle_of_word:[4,2,1,""],test_adjust_start_shifts_to_next_character:[4,2,1,""],test_adjust_start_shifts_to_previous_character:[4,2,1,""],test_adjust_start_shifts_to_second_previous_characters:[4,2,1,""],test_map_string_maps_mixed_string:[4,2,1,""],test_map_string_maps_space_string:[4,2,1,""],test_map_string_maps_word_string:[4,2,1,""]},"test.test_fuzzy_match.TestMatchInContext":{setUp:[4,2,1,""],test_context_contains_text_from_doc:[4,2,1,""],test_context_is_adjustable:[4,2,1,""],test_make_match_in_context:[4,2,1,""]},"test.test_fuzzy_phrase":{Test:[4,1,1,""],TestFuzzyPhrase:[4,1,1,""]},"test.test_fuzzy_phrase.Test":{test_skipgrams_have_correct_length:[4,2,1,""],test_text2skipgrams_accepts_positive_ngram_size:[4,2,1,""],test_text2skipgrams_rejects_negative_ngram_size:[4,2,1,""],test_text2skipgrams_rejects_negative_skip_size:[4,2,1,""]},"test.test_fuzzy_phrase.TestFuzzyPhrase":{test_fuzzy_phrase_accepts_phrase_as_dict:[4,2,1,""],test_fuzzy_phrase_accepts_phrase_as_string:[4,2,1,""],test_fuzzy_phrase_accepts_phrase_with_valid_list_of_strings_label:[4,2,1,""],test_fuzzy_phrase_accepts_phrase_with_valid_string_label:[4,2,1,""],test_fuzzy_phrase_can_set_max_end:[4,2,1,""],test_fuzzy_phrase_can_set_max_offset:[4,2,1,""],test_fuzzy_phrase_cannot_set_negative_max_offset:[4,2,1,""],test_fuzzy_phrase_rejects_phrase_with_invalid_label:[4,2,1,""]},"test.test_fuzzy_phrase_model":{Test:[4,1,1,""]},"test.test_fuzzy_phrase_model.Test":{test_can_add_custom_key_value_pairs_to_phrase:[4,2,1,""],test_can_add_distractors:[4,2,1,""],test_can_add_label_as_list_to_phrase:[4,2,1,""],test_can_add_label_to_phrase:[4,2,1,""],test_can_add_variant_phrase:[4,2,1,""],test_can_configure_ngram_size:[4,2,1,""],test_making_empty_phrase_model:[4,2,1,""],test_making_phrase_model_with_list_of_keyword_strings:[4,2,1,""],test_making_phrase_model_with_list_of_phrase_dictionaries:[4,2,1,""],test_phrase_model_can_add_phrase:[4,2,1,""],test_phrase_model_can_remove_phrase:[4,2,1,""],test_phrase_model_indexes_phrase_words:[4,2,1,""]},"test.test_fuzzy_phrase_searcher":{TestCandidate:[4,1,1,""],TestFuzzyPhraseSearcher:[4,1,1,""],TestFuzzySearchExactMatch:[4,1,1,""],TestSearcherRealData:[4,1,1,""],TestSkipMatches:[4,1,1,""]},"test.test_fuzzy_phrase_searcher.TestCandidate":{test_candidate_detects_no_match:[4,2,1,""],test_candidate_detects_no_match_with_no_skip_match:[4,2,1,""],test_candidate_has_skipgram_overlap:[4,2,1,""]},"test.test_fuzzy_phrase_searcher.TestFuzzyPhraseSearcher":{test_can_add_phrases_as_phrase_objects:[4,2,1,""],test_can_add_phrases_as_strings:[4,2,1,""],test_can_filter_skipgram_threshold:[4,2,1,""],test_can_generate_skip_matches:[4,2,1,""],test_can_get_candidates:[4,2,1,""],test_can_make_default_phrase_searcher:[4,2,1,""],test_finds_multiple_candidates:[4,2,1,""],test_searcher_can_match_distractors:[4,2,1,""],test_searcher_can_match_variants:[4,2,1,""],test_searcher_can_register_distractors:[4,2,1,""],test_searcher_can_register_variants:[4,2,1,""],test_searcher_can_toggle_distractors:[4,2,1,""],test_searcher_can_toggle_variants:[4,2,1,""],test_searcher_finds_correct_start:[4,2,1,""],test_searcher_finds_near_match:[4,2,1,""],test_searcher_finds_repeat_phrases_as_multiple_matches:[4,2,1,""],test_searcher_handles_ignorecase:[4,2,1,""],test_searcher_is_case_sensitive:[4,2,1,""],test_searcher_uses_word_boundaries:[4,2,1,""]},"test.test_fuzzy_phrase_searcher.TestFuzzySearchExactMatch":{setUp:[4,2,1,""],test_fuzzy_search_can_search_exact_match:[4,2,1,""],test_fuzzy_search_can_search_exact_match_with_special_characters:[4,2,1,""],test_fuzzy_search_can_search_exact_match_with_word_boundaries:[4,2,1,""],test_fuzzy_search_can_search_exact_match_without_word_boundaries:[4,2,1,""],test_text_split:[4,2,1,""]},"test.test_fuzzy_phrase_searcher.TestSearcherRealData":{setUp:[4,2,1,""],test_fuzzy_search_text1_finds_attendants:[4,2,1,""],test_fuzzy_search_text1_finds_date:[4,2,1,""],test_fuzzy_search_text1_finds_four_matches:[4,2,1,""],test_fuzzy_search_text1_finds_friday:[4,2,1,""],test_fuzzy_search_text1_finds_president:[4,2,1,""],test_fuzzy_search_text2_finds_attendants:[4,2,1,""],test_fuzzy_search_text2_finds_date:[4,2,1,""],test_fuzzy_search_text2_finds_four_matches:[4,2,1,""],test_fuzzy_search_text2_finds_friday:[4,2,1,""],test_fuzzy_search_text2_finds_president:[4,2,1,""]},"test.test_fuzzy_phrase_searcher.TestSkipMatches":{test_skip_matches_registers_match:[4,2,1,""]},"test.test_fuzzy_string":{Test:[4,1,1,""]},"test.test_fuzzy_string.Test":{test_make_ngrams_accepts_positive_integer:[4,2,1,""],test_make_ngrams_handles_size_one_correctly:[4,2,1,""],test_make_ngrams_handles_size_two_correctly:[4,2,1,""],test_make_ngrams_rejects_integer_larger_than_text_length:[4,2,1,""],test_make_ngrams_rejects_negative_size:[4,2,1,""],test_make_ngrams_rejects_non_integer_size:[4,2,1,""],test_make_ngrams_rejects_non_string_text:[4,2,1,""],test_score_char_overlap_with_self_is_len_of_self:[4,2,1,""],test_score_char_overlap_with_smaller_word_is_smaller_than_len_of_self:[4,2,1,""],test_score_levenshtein_similarity_with_self_is_len_of_self:[4,2,1,""],test_score_ngram_overlap_is_num_ngrams_for_self_comparison:[4,2,1,""],test_score_ngram_overlap_is_zero_for_comparison_with_empty:[4,2,1,""]},"test.test_fuzzy_template":{TestFuzzyTemplate:[4,1,1,""],TestFuzzyTemplateElement:[4,1,1,""],TestFuzzyTemplateGroup:[4,1,1,""],TestFuzzyTemplateWithRealData:[4,1,1,""]},"test.test_fuzzy_template.TestFuzzyTemplate":{setUp:[4,2,1,""],test_template_can_get_phrase_by_label:[4,2,1,""],test_template_can_ignore_element_with_unknown_label:[4,2,1,""],test_template_can_register_group_elements:[4,2,1,""],test_template_can_return_required_elements:[4,2,1,""],test_template_cannot_register_element_with_unknown_label:[4,2,1,""],test_template_generation:[4,2,1,""],test_template_get_phrase_by_label_returns_correct_phrase:[4,2,1,""],test_template_get_phrase_by_label_returns_phrase_object:[4,2,1,""],test_template_register_simple_element:[4,2,1,""],test_template_register_simple_element_as_multi_if_no_cardinality:[4,2,1,""],test_template_register_simple_element_with_list_labels:[4,2,1,""],test_template_returns_all_required_element_labels:[4,2,1,""],test_template_returns_all_required_elements:[4,2,1,""]},"test.test_fuzzy_template.TestFuzzyTemplateElement":{test_template_accepts_label_and_cardinality:[4,2,1,""],test_template_accepts_label_only:[4,2,1,""],test_template_rejects_invalid_cardinality_value:[4,2,1,""]},"test.test_fuzzy_template.TestFuzzyTemplateGroup":{setUp:[4,2,1,""],test_template_group_accepts_label_and_order:[4,2,1,""],test_template_group_accepts_label_only:[4,2,1,""]},"test.test_fuzzy_template.TestFuzzyTemplateWithRealData":{setUp:[4,2,1,""],test_template_can_read_real_data:[4,2,1,""]},"test.test_fuzzy_template_searcher":{TestFuzzyTemplateSearcher:[4,1,1,""],TestFuzzyTemplateSearcherWithRealData:[4,1,1,""]},"test.test_fuzzy_template_searcher.TestFuzzyTemplateSearcher":{setUp:[4,2,1,""],test_add_template_sets_phrase_model:[4,2,1,""],test_can_add_template_at_init:[4,2,1,""],test_can_add_template_later:[4,2,1,""],test_can_make_searcher:[4,2,1,""],test_can_search_text:[4,2,1,""],test_configure_ngram_size:[4,2,1,""],test_search_text_returns_template_matches:[4,2,1,""],test_throws_error_for_mismatch_ngram_size:[4,2,1,""]},"test.test_fuzzy_template_searcher.TestFuzzyTemplateSearcherWithRealData":{prep_test:[4,2,1,""],setUp:[4,2,1,""],test_search_text_finds_template_with_auction_test_1:[4,2,1,""],test_search_text_finds_template_with_auction_test_2:[4,2,1,""],test_search_text_finds_template_with_auction_test_3:[4,2,1,""],test_search_text_finds_template_with_auction_test_4:[4,2,1,""]},fuzzy_search:{fuzzy_config:[1,0,0,"-"],fuzzy_context_searcher:[1,0,0,"-"],fuzzy_match:[1,0,0,"-"],fuzzy_patterns:[1,0,0,"-"],fuzzy_phrase:[1,0,0,"-"],fuzzy_phrase_model:[1,0,0,"-"],fuzzy_phrase_searcher:[1,0,0,"-"],fuzzy_searcher:[1,0,0,"-"],fuzzy_string:[1,0,0,"-"],fuzzy_template:[1,0,0,"-"],fuzzy_template_searcher:[1,0,0,"-"]},test:{test_fuzzy_context_searcher:[4,0,0,"-"],test_fuzzy_match:[4,0,0,"-"],test_fuzzy_phrase:[4,0,0,"-"],test_fuzzy_phrase_model:[4,0,0,"-"],test_fuzzy_phrase_searcher:[4,0,0,"-"],test_fuzzy_string:[4,0,0,"-"],test_fuzzy_template:[4,0,0,"-"],test_fuzzy_template_searcher:[4,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"boolean":1,"case":[0,1,4],"char":1,"class":[1,4],"default":1,"float":1,"function":1,"int":1,"long":1,"new":1,"return":1,"true":1,For:1,The:[0,1],There:1,Use:1,add:1,add_custom:1,add_distractor:1,add_label:1,add_match_context:1,add_max_offset:1,add_metadata:1,add_model:1,add_new_phras:1,add_phras:1,add_scor:1,add_skip_match:1,add_vari:1,added:1,adding:1,addit:1,adjust:1,adjust_match_end_offset:1,adjust_match_offset:1,adjust_match_start_offset:1,affix:1,affix_str:1,against:1,all:1,allow:[0,1],allow_overlapping_match:1,also:1,ani:[0,1],annot:1,anoth:1,appear:1,approxim:0,archiv:0,around:1,arrai:1,as_phrase_object:1,as_web_anno:1,base:[1,4],been:[0,1],befor:4,belong:1,better:1,between:1,big:1,bool:1,boundari:1,calcul:1,calculate_end_shift:1,can:1,candid:1,candidate_end_offset:1,candidate_start_offset:1,candidate_str:1,candidates_to_match:1,cardin:1,char_match_threshold:1,charact:1,check:1,check_entry_phras:[],closer:1,collect:0,come:[],common:1,compar:1,comput:1,conduct:1,config:1,configur:1,configure_context:1,connect:1,contain:[0,1],content:[0,2],context:1,context_before_pattern:1,context_pattern:1,context_s:1,context_str:1,context_then_pattern:1,context_typ:1,copi:1,correspond:1,count:1,creat:0,create_term_match:1,custom:1,custom_properti:1,deriv:1,determin:1,develop:0,deviat:1,dict:1,dictionari:1,differ:1,difficult:0,digit:0,disable_strip_suffix:1,distanc:1,distractor:1,distractor_phras:1,distractors_of_phras:1,document:1,doesn:1,each:1,earli:1,early_threshold:1,either:1,element:[0,1],element_info:1,element_label:1,element_start_index:1,element_typ:1,enable_strip_suffix:1,end:1,end_index:1,end_offset:1,entir:1,entri:1,error:0,escape_str:1,exact:1,exact_match:1,exercis:4,extend:1,fals:1,filter:1,filter_candid:1,filter_char_match_candid:1,filter_distractor:1,filter_levenshtein_candid:1,filter_matches_by_distractor:1,filter_matches_by_threshold:1,filter_ngram_candid:1,filter_overlapping_phrase_candid:1,filter_phrase_match:1,filter_skipgram_threshold:1,find:[0,1],find_candid:1,find_exact_match:1,find_match:1,find_matches_in_context:1,find_next_element_end_index:1,find_next_element_start_index:1,find_next_group_match_sequ:1,find_next_ordered_group_match_sequ:1,find_next_unordered_group_match_sequ:1,find_skipgram_match:1,find_start_candid:1,find_template_match:1,find_term_match:1,first:1,fit:1,fixtur:4,flag:1,fraction:1,from:1,fuzzi:1,fuzzy_config:[0,2],fuzzy_context_search:[0,2],fuzzy_match:[0,2],fuzzy_pattern:[0,2],fuzzy_phras:[0,2],fuzzy_phrase_model:[0,2],fuzzy_phrase_search:[0,2],fuzzy_search:[0,2],fuzzy_str:[0,2],fuzzy_templ:[0,2],fuzzy_template_search:[0,2],fuzzycontextsearch:1,fuzzyphrasesearch:1,fuzzysearch:1,fuzzytempl:1,fuzzytemplateel:1,fuzzytemplategroupel:1,fuzzytemplatelabelel:1,fuzzytemplatesearch:1,gener:1,generate_group_from_json:1,generate_label_from_json:1,get:1,get_context_pattern:1,get_el:1,get_elements_by_cardin:1,get_exact_match_rang:1,get_known_word_offset:1,get_label:1,get_label_el:[],get_label_phras:1,get_labels_by_cardin:1,get_match_start_offset:1,get_match_str:1,get_non_word_prefix:1,get_non_word_suffix:1,get_phras:1,get_phrase_match_list_label:1,get_phrases_by_max_length:1,get_required_el:1,get_required_label:1,get_search_pattern:1,get_sequence_label_element_match:1,get_skip_count_overlap:1,get_skip_set_overlap:1,get_skipmatch_candid:1,get_skipmatch_phrase_candid:1,get_skipset_overlap:1,get_text_dict:1,get_vari:1,given:1,goe:1,gram:1,group:1,group_el:1,group_info:1,has:[0,1],has_custom:1,has_group:1,has_label:1,has_phras:1,has_required_match:1,has_skipgram:1,have:1,histor:0,hook:4,how:1,htr:0,hundr:0,identifi:1,ignor:1,ignore_cas:1,ignore_unknown:1,ignorecas:1,includ:1,include_vari:1,incorrect:1,index:[0,1],index_distractor:1,index_known_word_offset:1,index_phras:1,index_phrase_model:1,index_phrase_word:1,index_vari:1,indic:1,individu:1,inf:1,initi:1,initialize_sequ:1,input:1,insert_skip:1,instead:1,integ:1,is_early_skipgram:1,is_label:1,is_match:1,is_phrase_dict:1,is_valid_label:1,iter:1,its:1,json:1,kei:1,keyphras:1,keyword:[0,1],known:1,known_word_offset:1,label:1,label_info:1,label_str:1,languag:0,late_threshold:1,later:1,least:1,length:[0,1],levenshtein:1,levenshtein_threshold:1,librari:0,like:1,list:[0,1],list_context_pattern_typ:1,list_pattern_definit:1,list_pattern_nam:1,local:1,longer:1,look:1,main:1,main_phras:1,make:0,make_ngram:1,make_search_context_pattern:1,mani:[0,1],map_str:1,match:[0,1],match_end:1,match_in_context:1,match_offset:1,match_phras:1,match_rang:1,match_str:1,match_term:1,match_vari:1,matchincontext:1,max_dist:1,max_length:1,max_length_vari:1,max_offset:1,maximum:1,meet:1,metadata:1,metadata_dict:1,method:[1,4],methodnam:4,million:0,model:1,modul:[0,2],more:1,multi:1,multipl:1,must:1,name:1,name_onli:1,next:1,next_distance_max:1,next_label:1,ngram:1,ngram_siz:1,ngram_threshold:1,non:1,none:[1,4],number:1,object1:1,object2:1,object:1,occur:1,occurr:1,ocr:0,offset:1,onc:1,one:1,onli:1,option:1,order:1,other:1,out:1,overlap:1,overrid:1,packag:[0,2],page:0,pair:1,param:1,paramet:1,pars:1,parse_group_el:1,parse_label_el:1,part:1,pass:1,pattern_before_context:1,pattern_comma_then_context:1,pattern_definit:1,pattern_nam:1,pattern_typ:1,phrase:[0,1],phrase_candid:1,phrase_dict:1,phrase_end:1,phrase_label:1,phrase_match:1,phrase_model:1,phrase_str:1,phrasematch:1,phrasematchincontext:1,phrasemodel:1,point:1,pre:1,prefix:1,prefix_s:1,prep_test:4,process:1,properti:1,proport:1,punctuat:1,python:0,rang:1,rank_candid:1,re_match:1,recognit:0,refer:1,regist:1,register_el:[],register_templ:1,remov:1,remove_custom:1,remove_distractor:1,remove_first_skip:1,remove_label:1,remove_phras:1,remove_phrase_word:1,remove_vari:1,repetit:0,represent:1,requir:1,respresent:1,result:1,routin:1,rtype:[],runtest:4,same:1,same_candid:1,score:1,score_char_overlap:1,score_char_overlap_ratio:1,score_character_overlap:1,score_levenshtein_dist:1,score_levenshtein_distance_ratio:1,score_levenshtein_similar:1,score_levenshtein_similarity_ratio:1,score_ngram_overlap:1,score_ngram_overlap_ratio:1,search:1,search_exact:1,search_exact_phras:1,search_exact_phrases_with_word_boundari:1,search_exact_phrases_without_word_boundari:1,search_text:1,searcher:1,searcher_config:[],second:1,sequenc:1,set:[1,4],set_label:1,set_strip_suffix:1,set_templ:1,setup:[2,4],share:1,share_label:1,shift_start_skip:1,should:1,similar:1,simpl:0,singl:1,size:1,skip:1,skip_exact_match:1,skip_gram:1,skip_match:1,skip_match_length:1,skip_siz:1,skip_threshold:1,skipgram1:1,skipgram2:1,skipgram:1,skipgram_combin:1,skipgram_length:1,skipgram_offset:1,skipgram_overlap:1,skipgram_str:1,skipgram_threshold:1,skipmatch:1,some:[0,1],sourc:1,specif:1,speed:1,spell:[0,1],start:1,start_index:1,step:1,stop:1,str:1,string:1,strip:1,strip_prefix:1,strip_suffix:1,submodul:[0,2],suffix:1,suffix_s:1,surround:1,taken:1,templat:1,template_el:1,template_group:1,template_json:1,template_sequ:1,template_start_index:1,templatematch:1,term1:1,term2:1,term:1,test:[1,2],test_add_template_sets_phrase_model:4,test_adjust_boundaries_finds_word_boundari:4,test_adjust_boundaries_removes_surrounding_whitespac:4,test_adjust_end:4,test_adjust_end_does_not_shift_when_end_middle_of_next_word:4,test_adjust_end_shifts_back_one_when_ending_with_whitespac:4,test_adjust_end_shifts_back_one_when_phrase_ends_with_whitespac:4,test_adjust_end_shifts_back_two_when_ending_with_whitespace_and_char:4,test_adjust_end_shifts_to_end_of_next_word:4,test_adjust_start:4,test_adjust_start_does_not_shift_to_third_previous_charact:4,test_adjust_start_returns_when_in_middle_of_word:4,test_adjust_start_shifts_to_next_charact:4,test_adjust_start_shifts_to_previous_charact:4,test_adjust_start_shifts_to_second_previous_charact:4,test_can_add_custom_key_value_pairs_to_phras:4,test_can_add_distractor:4,test_can_add_label_as_list_to_phras:4,test_can_add_label_to_phras:4,test_can_add_phrases_as_phrase_object:4,test_can_add_phrases_as_str:4,test_can_add_template_at_init:4,test_can_add_template_lat:4,test_can_add_variant_phras:4,test_can_configure_ngram_s:4,test_can_filter_skipgram_threshold:4,test_can_generate_skip_match:4,test_can_get_candid:4,test_can_make_default_phrase_search:4,test_can_make_search:4,test_can_search_text:4,test_candidate_detects_no_match:4,test_candidate_detects_no_match_with_no_skip_match:4,test_candidate_has_skipgram_overlap:4,test_configure_ngram_s:4,test_context_contains_text_from_doc:4,test_context_is_adjust:4,test_data:[],test_finds_multiple_candid:4,test_fuzzy_context_search:2,test_fuzzy_context_searcher_can_add_context:4,test_fuzzy_context_searcher_can_search_context:4,test_fuzzy_context_searcher_can_set_context_s:4,test_fuzzy_context_searcher_finds_match_with_context:4,test_fuzzy_match:2,test_fuzzy_phras:2,test_fuzzy_phrase_accepts_phrase_as_dict:4,test_fuzzy_phrase_accepts_phrase_as_str:4,test_fuzzy_phrase_accepts_phrase_with_valid_list_of_strings_label:4,test_fuzzy_phrase_accepts_phrase_with_valid_string_label:4,test_fuzzy_phrase_can_set_max_end:4,test_fuzzy_phrase_can_set_max_offset:4,test_fuzzy_phrase_cannot_set_negative_max_offset:4,test_fuzzy_phrase_model:2,test_fuzzy_phrase_rejects_phrase_with_invalid_label:4,test_fuzzy_phrase_search:2,test_fuzzy_search_can_search_exact_match:4,test_fuzzy_search_can_search_exact_match_with_special_charact:4,test_fuzzy_search_can_search_exact_match_with_word_boundari:4,test_fuzzy_search_can_search_exact_match_without_word_boundari:4,test_fuzzy_search_text1_finds_attend:4,test_fuzzy_search_text1_finds_d:4,test_fuzzy_search_text1_finds_four_match:4,test_fuzzy_search_text1_finds_fridai:4,test_fuzzy_search_text1_finds_presid:4,test_fuzzy_search_text2_finds_attend:4,test_fuzzy_search_text2_finds_d:4,test_fuzzy_search_text2_finds_four_match:4,test_fuzzy_search_text2_finds_fridai:4,test_fuzzy_search_text2_finds_presid:4,test_fuzzy_str:2,test_fuzzy_templ:2,test_fuzzy_template_search:2,test_make_match_in_context:4,test_make_ngrams_accepts_positive_integ:4,test_make_ngrams_handles_size_one_correctli:4,test_make_ngrams_handles_size_two_correctli:4,test_make_ngrams_rejects_integer_larger_than_text_length:4,test_make_ngrams_rejects_negative_s:4,test_make_ngrams_rejects_non_integer_s:4,test_make_ngrams_rejects_non_string_text:4,test_making_empty_phrase_model:4,test_making_phrase_model_with_list_of_keyword_str:4,test_making_phrase_model_with_list_of_phrase_dictionari:4,test_map_string_maps_mixed_str:4,test_map_string_maps_space_str:4,test_map_string_maps_word_str:4,test_nam:4,test_phrase_model_can_add_phras:4,test_phrase_model_can_remove_phras:4,test_phrase_model_indexes_phrase_word:4,test_score_char_overlap_with_self_is_len_of_self:4,test_score_char_overlap_with_smaller_word_is_smaller_than_len_of_self:4,test_score_levenshtein_similarity_with_self_is_len_of_self:4,test_score_ngram_overlap_is_num_ngrams_for_self_comparison:4,test_score_ngram_overlap_is_zero_for_comparison_with_empti:4,test_search_text_finds_template_with_auction_test_1:4,test_search_text_finds_template_with_auction_test_2:4,test_search_text_finds_template_with_auction_test_3:4,test_search_text_finds_template_with_auction_test_4:4,test_search_text_returns_template_match:4,test_searcher_can_match_distractor:4,test_searcher_can_match_vari:4,test_searcher_can_register_distractor:4,test_searcher_can_register_vari:4,test_searcher_can_toggle_distractor:4,test_searcher_can_toggle_vari:4,test_searcher_finds_correct_start:4,test_searcher_finds_near_match:4,test_searcher_finds_repeat_phrases_as_multiple_match:4,test_searcher_handles_ignorecas:4,test_searcher_is_case_sensit:4,test_searcher_uses_word_boundari:4,test_skip_matches_registers_match:4,test_skipgrams_have_correct_length:4,test_template_accepts_label_and_cardin:4,test_template_accepts_label_onli:4,test_template_can_get_phrase_by_label:4,test_template_can_ignore_element_with_unknown_label:4,test_template_can_read_real_data:4,test_template_can_register_group_el:4,test_template_can_return_required_el:4,test_template_cannot_register_element_with_unknown_label:4,test_template_gener:4,test_template_get_phrase_by_label_returns_correct_phras:4,test_template_get_phrase_by_label_returns_phrase_object:4,test_template_group_accepts_label_and_ord:4,test_template_group_accepts_label_onli:4,test_template_register_simple_el:4,test_template_register_simple_element_as_multi_if_no_cardin:4,test_template_register_simple_element_with_list_label:4,test_template_rejects_invalid_cardinality_valu:4,test_template_returns_all_required_el:4,test_template_returns_all_required_element_label:4,test_text2skipgrams_accepts_positive_ngram_s:4,test_text2skipgrams_rejects_negative_ngram_s:4,test_text2skipgrams_rejects_negative_skip_s:4,test_text_split:4,test_throws_error_for_mismatch_ngram_s:4,testcandid:4,testcas:4,testfuzzycontextsearch:4,testfuzzymatch:4,testfuzzyphras:4,testfuzzyphrasesearch:4,testfuzzysearchexactmatch:4,testfuzzytempl:4,testfuzzytemplateel:4,testfuzzytemplategroup:4,testfuzzytemplatesearch:4,testfuzzytemplatesearcherwithrealdata:4,testfuzzytemplatewithrealdata:4,testmatchincontext:4,testsearcherrealdata:4,testskipmatch:4,text2skipgram:1,text:[0,1],text_doc:1,text_id:1,text_suffix:1,than:1,them:1,thi:1,those:1,thousand:0,threshold:1,time:1,to_json:[],togeth:1,toggl:1,token:1,too:1,treat:1,tupl:1,turn:1,two:1,type:1,typic:0,under:1,union:1,unittest:4,unknown:1,unord:1,updat:1,use:[0,1],use_word_boundari:1,used:1,valid:1,validate_element_properti:1,validate_entry_phras:1,validate_match_prop:1,valu:1,variabl:1,variant:1,variant_of:1,variant_phras:1,variants_of_phras:1,variat:0,w3c:1,web:1,well:1,what:1,when:1,where:[0,1],whether:1,which:1,whitespac:1,whitespace_onli:1,window:1,within_rang:1,within_range_threshold:1,word:1,yield:1,you:[0,1]},titles:["Welcome to fuzzy-search\u2019s documentation!","fuzzy_search package","fuzzy-search","setup module","test package"],titleterms:{content:[1,4],document:0,fuzzi:[0,2],fuzzy_config:1,fuzzy_context_search:1,fuzzy_match:1,fuzzy_pattern:1,fuzzy_phras:1,fuzzy_phrase_model:1,fuzzy_phrase_search:1,fuzzy_search:1,fuzzy_str:1,fuzzy_templ:1,fuzzy_template_search:1,indic:0,modul:[1,3,4],packag:[1,4],search:[0,2],setup:3,submodul:[1,4],tabl:0,test:4,test_data:[],test_fuzzy_context_search:4,test_fuzzy_match:4,test_fuzzy_phras:4,test_fuzzy_phrase_model:4,test_fuzzy_phrase_search:4,test_fuzzy_str:4,test_fuzzy_templ:4,test_fuzzy_template_search:4,welcom:0}})