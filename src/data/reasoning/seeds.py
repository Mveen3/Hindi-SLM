"""
Seed Phrase Banks (hand-authored, Hindi and Nepali)
===================================================

A complete, hand-written phrase bank for each language: entity names,
attribute vocabulary, and at least two grammatical variants of every template
slot declared in ``tasks.py``.

These seeds serve three purposes.

1. **Reproducibility.**  ``python main.py reasoning-data --offline``
   regenerates the entire finetuning corpus with no API key and no network,
   which is what a grader without our Groq key needs.
2. **Few-shot anchors.**  Each seed template is shown to ``gpt-oss-120b`` as a
   worked example of its slot, which is why the model returns templates that
   satisfy the placeholder contract instead of free prose.
3. **Fallback.**  Any LLM template that fails validation is simply dropped; the
   seeds guarantee every slot still has enough variants to split across
   train / val / test.

Grammatical notes
-----------------
Hindi possessives, copulas and interrogatives agree with the *attribute noun's*
gender, so those words are attributes of the noun (``poss``, ``whose``,
``howmuch``, ``copula``) rather than constants.  Comparisons are anchored on
the attribute ("X's age is more than Y's age") rather than on the person
("X is older"), because the latter would need the *person's* gender, which a
name alone does not give.  Nepali takes the invariant ``को`` / ``कसको`` /
``छ``, and its templates write ``{A}{poss}`` with no space, as the postposition
is written joined to the name.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hindi (Model H)
# ---------------------------------------------------------------------------

HINDI_SEED: dict = {
    "labels": {"question": "प्रश्न", "reasoning": "तर्क", "answer": "उत्तर"},
    "answers": {"yes": "हाँ", "no": "नहीं"},

    "entities": {
        "people": [
            "राम", "श्याम", "मोहन", "सोहन", "गीता", "सीता", "अनिल", "सुनील",
            "कविता", "रेखा", "अजय", "विजय", "प्रीति", "नेहा", "राहुल", "संजय",
            "दीपक", "मीना", "आशा", "किरण", "पूजा", "मनोज", "सुरेश", "रमेश",
            "अंजलि", "स्नेहा", "विकास", "गौरव", "शालिनी", "दिव्या",
        ],
        "cities": [
            "दिल्ली", "मुंबई", "कोलकाता", "चेन्नई", "जयपुर", "लखनऊ", "पटना",
            "भोपाल", "इंदौर", "नागपुर", "कानपुर", "वाराणसी", "अहमदाबाद", "सूरत",
            "रांची", "रायपुर",
            "बेंगलुरु", "हैदराबाद", "पुणे", "अमृतसर", "देहरादून", "गुवाहाटी", "मैसूर", "उदयपुर",],
        "objects": [
            "किताब", "कुर्सी", "मेज़", "घड़ी", "कलम", "बैग", "साइकिल", "पंखा",
            "छाता", "जूता", "दर्पण", "बोतल", "टोकरी", "चादर", "तकिया", "थाली",
            "चश्मा", "तराजू", "बाल्टी", "पेटी", "रस्सी", "टोपी", "चम्मच", "कंघी",],
        "animals": [
            "शेर", "हाथी", "घोड़ा", "हिरण", "बाघ", "चीता", "ऊँट", "बंदर",
            "खरगोश", "कुत्ता", "भालू", "लोमड़ी",
            "गाय", "भैंस", "बकरी", "बिल्ली", "गधा", "सियार", "नेवला", "गिलहरी", "मगरमच्छ", "जिराफ", "गैंडा", "जेब्रा",],
        "fruits": [
            "आम", "केला", "सेब", "अंगूर", "संतरा", "अमरूद", "पपीता", "अनार",
            "तरबूज़", "नाशपाती", "लीची", "आलूबुखारा",
            "खरबूजा", "चीकू", "कटहल", "शहतूत", "जामुन", "बेर", "अंजीर", "खुबानी", "नारियल", "सीताफल", "मौसमी", "कीवी",],
    },

    "attributes": {
        "age": {"noun": "उम्र", "poss": "की", "poss_obl": "की", "whose": "किसकी", "howmuch": "कितनी",
                "unit": "साल", "copula": "है", "more": "अधिक", "less": "कम"},
        "height": {"noun": "ऊँचाई", "poss": "की", "poss_obl": "की", "whose": "किसकी", "howmuch": "कितनी",
                   "unit": "सेंटीमीटर", "copula": "है", "more": "अधिक", "less": "कम"},
        "weight": {"noun": "वज़न", "poss": "का", "poss_obl": "के", "whose": "किसका", "howmuch": "कितना",
                   "unit": "किलोग्राम", "copula": "है", "more": "अधिक", "less": "कम"},
        "price": {"noun": "कीमत", "poss": "की", "poss_obl": "की", "whose": "किसकी", "howmuch": "कितनी",
                  "unit": "रुपये", "copula": "है", "more": "अधिक", "less": "कम"},
        "quantity": {"noun": "मात्रा", "poss": "की", "poss_obl": "की", "whose": "किसकी", "howmuch": "कितनी",
                     "unit": "किलोग्राम", "copula": "है", "more": "अधिक", "less": "कम"},
        "distance": {"noun": "दूरी", "poss": "की", "poss_obl": "की", "whose": "किसकी", "howmuch": "कितनी",
                     "unit": "किलोमीटर", "copula": "है", "more": "अधिक", "less": "कम"},
        "population": {"noun": "जनसंख्या", "poss": "की", "poss_obl": "की", "whose": "किसकी", "howmuch": "कितनी",
                       "unit": "लाख", "copula": "है", "more": "अधिक", "less": "कम"},
        "speed": {"noun": "गति", "poss": "की", "poss_obl": "की", "whose": "किसकी", "howmuch": "कितनी",
                  "unit": "किलोमीटर प्रति घंटा", "copula": "है", "more": "अधिक", "less": "कम"},
        "temperature": {"noun": "तापमान", "poss": "का", "poss_obl": "के", "whose": "किसका", "howmuch": "कितना",
                        "unit": "डिग्री सेल्सियस", "copula": "है", "more": "अधिक", "less": "कम"},
    },

    "templates": {
        "fact": [
            "{E} {poss} {noun} {v} {unit} {copula}।",
            "दी गई जानकारी के अनुसार {E} {poss} {noun} {v} {unit} {copula}।",
            "यह ज्ञात है कि {E} {poss} {noun} {v} {unit} {copula}।",
            "{E} {poss} {noun} ठीक {v} {unit} {copula}।",
        ],
        "rel_fact": [
            "{A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} {copula}।",
            "दी गई जानकारी के अनुसार {A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} {copula}।",
            "यह ज्ञात है कि {A} {poss} {noun}, {B} {poss_obl} {noun} की तुलना में {cmp} {copula}।",
        ],
        "q_which_of_two": [
            "{A} और {B} में {whose} {noun} {cmp} है?",
            "{A} और {B} की तुलना कीजिए: {whose} {noun} {cmp} है?",
            "बताइए कि {A} और {B} में {whose} {noun} {cmp} है?",
            "उपरोक्त जानकारी के आधार पर {A} और {B} में {whose} {noun} {cmp} है?",
        ],
        "q_extreme": [
            "इनमें {whose} {noun} सबसे {cmp} है?",
            "उपरोक्त में से {whose} {noun} सबसे {cmp} है?",
            "बताइए कि इन सभी में {whose} {noun} सबसे {cmp} है?",
            "दी गई जानकारी के आधार पर {whose} {noun} सबसे {cmp} है?",
        ],
        "q_equality": [
            "क्या {A} और {B} {poss} {noun} बराबर है?",
            "क्या {A} {poss} {noun} और {B} {poss} {noun} समान है?",
            "बताइए कि {A} और {B} {poss} {noun} एक समान है या नहीं?",
            "क्या इन दोनों, {A} और {B}, {poss} {noun} बराबर है?",
        ],
        "q_difference": [
            "{A} {poss} {noun}, {B} {poss_obl} {noun} से {howmuch} {cmp} है?",
            "बताइए कि {A} {poss} {noun}, {B} {poss_obl} {noun} से {howmuch} {cmp} है?",
            "{A} और {B} {poss} {noun} का अंतर क्या है?",
            "उपरोक्त जानकारी के आधार पर {A} {poss} {noun}, {B} {poss_obl} {noun} से {howmuch} {cmp} है?",
        ],
        "q_ordering": [
            "इन सभी को {noun} के {cmp_from} से {cmp_to} क्रम में लिखिए।",
            "उपरोक्त सभी को {noun} के अनुसार {cmp_from} से {cmp_to} क्रम में व्यवस्थित कीजिए।",
            "इन्हें {noun} के आधार पर {cmp_from} से {cmp_to} क्रम में बताइए।",
            "दी गई जानकारी के अनुसार इन सभी को {noun} के {cmp_from} से {cmp_to} क्रम में लगाइए।",
        ],
        "q_middle": [
            "इन तीनों में {whose} {noun} न सबसे अधिक है और न सबसे कम?",
            "{noun} के आधार पर इन तीनों में बीच में कौन है?",
            "इन तीनों में {whose} {noun} मध्य में है?",
            "दी गई जानकारी के अनुसार {noun} के क्रम में बीच में कौन है?",
        ],
        "q_verify": [
            "क्या यह सही है कि {A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} है?",
            "बताइए कि यह कथन सही है या नहीं: {A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} है।",
            "क्या {A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} है?",
            "उपरोक्त जानकारी के आधार पर बताइए कि {A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} है या नहीं?",
        ],

        "fact_step": [
            "{E} {poss} {noun} = {v} {unit}।",
            "{E}: {noun} {v} {unit}।",
        ],
        "rel_step": [
            "{A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} है।",
            "तुलना करने पर {A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} है।",
        ],
        "deduce_step": [
            "चूँकि {A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp} है और {B} {poss} {noun}, {C} {poss_obl} {noun} से {cmp} है, इसलिए {A} {poss} {noun}, {C} {poss_obl} {noun} से {cmp} है।",
            "{A} से {B} और {B} से {C} की श्रृंखला जोड़ने पर {A} {poss} {noun}, {C} {poss_obl} {noun} से {cmp} है।",
        ],
        "flip_step": [
            "{A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp_from} है, अर्थात् {B} {poss} {noun}, {A} {poss_obl} {noun} से {cmp_to} है।",
            "इसे उलटकर लिखें: {A} {poss} {noun}, {B} {poss_obl} {noun} से {cmp_from} है, तो {B} {poss} {noun}, {A} {poss_obl} {noun} से {cmp_to} है।",
        ],
        "diff_step": [
            "{vA} - {vB} = {d} {unit}।",
            "अंतर निकालने पर {vA} {unit} में से {vB} {unit} घटाने पर {d} {unit} बचता है।",
        ],
        "order_step": [
            "{noun} का क्रम इस प्रकार है: {ordered}।",
            "क्रम में लगाने पर: {ordered}।",
        ],
        "equal_step": [
            "{A} और {B} {poss} {noun} दोनों {v} {unit} है, अर्थात् बराबर है।",
            "दोनों {poss} {noun} समान है: {v} {unit}।",
        ],
        "conclusion_step": [
            "अतः उत्तर {answer} है।",
            "इसलिए उत्तर {answer} है।",
        ],
    },
}


# ---------------------------------------------------------------------------
# Nepali (Model L)
# ---------------------------------------------------------------------------

NEPALI_SEED: dict = {
    "labels": {"question": "प्रश्न", "reasoning": "तर्क", "answer": "उत्तर"},
    "answers": {"yes": "हो", "no": "होइन"},

    "entities": {
        "people": [
            "राम", "हरि", "सीता", "गीता", "कृष्ण", "बिनोद", "सुनिता", "कमला",
            "दीपक", "अनिल", "सरिता", "मनोज", "रमेश", "प्रकाश", "निर्मला", "शान्ति",
            "सागर", "लक्ष्मी", "पुष्पा", "नरेश", "सुरेश", "माया", "बबिता", "गोपाल",
            "सन्तोष", "रेखा", "उमेश", "कल्पना", "धनराज", "मीना",
        ],
        "cities": [
            "काठमाडौँ", "पोखरा", "विराटनगर", "ललितपुर", "भरतपुर", "बिरगंज",
            "धरान", "नेपालगंज", "जनकपुर", "हेटौडा", "बुटवल", "धनगढी",
            "इटहरी", "त्रिभुवननगर", "भक्तपुर", "दमक",
            "राजबिराज", "गोरखा", "बाग्लुङ", "तानसेन", "इलाम", "सुर्खेत", "डोटी", "बेसीसहर",],
        "objects": [
            "किताब", "कुर्सी", "टेबल", "घडी", "कलम", "झोला", "साइकल", "पंखा",
            "छाता", "जुत्ता", "ऐना", "बोतल", "डालो", "ओछ्यान", "तकिया", "थाल",
            "चस्मा", "तराजु", "बाल्टिन", "पेटी", "डोरी", "टोपी", "चम्चा", "काइँयो",],
        "animals": [
            "सिंह", "हात्ती", "घोडा", "मृग", "बाघ", "चितुवा", "ऊँट", "बाँदर",
            "खरायो", "कुकुर", "भालु", "स्याल",
            "गाई", "भैंसी", "बाख्रा", "बिरालो", "गधा", "न्याउरी", "लोखर्के", "गोही", "जिराफ", "गैंडा", "जेब्रा",],
        "fruits": [
            "आँप", "केरा", "स्याउ", "अंगुर", "सुन्तला", "अम्बा", "मेवा", "अनार",
            "खरबुजा", "नासपाती", "लिची", "आरुबखडा",
            "मेहल", "रुखकटहर", "किम्बु", "जामुन", "बयर", "अन्जीर", "खुर्पानी", "नरिवल", "सरिफा", "मौसमी", "किवी",],
    },

    "attributes": {
        "age": {"noun": "उमेर", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                "unit": "वर्ष", "copula": "छ", "more": "धेरै", "less": "कम"},
        "height": {"noun": "उचाइ", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                   "unit": "सेन्टिमिटर", "copula": "छ", "more": "धेरै", "less": "कम"},
        "weight": {"noun": "तौल", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                   "unit": "किलोग्राम", "copula": "छ", "more": "धेरै", "less": "कम"},
        "price": {"noun": "मूल्य", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                  "unit": "रुपैयाँ", "copula": "छ", "more": "धेरै", "less": "कम"},
        "quantity": {"noun": "परिमाण", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                     "unit": "किलोग्राम", "copula": "छ", "more": "धेरै", "less": "कम"},
        "distance": {"noun": "दूरी", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                     "unit": "किलोमिटर", "copula": "छ", "more": "धेरै", "less": "कम"},
        "population": {"noun": "जनसंख्या", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                       "unit": "लाख", "copula": "छ", "more": "धेरै", "less": "कम"},
        "speed": {"noun": "गति", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                  "unit": "किलोमिटर प्रति घण्टा", "copula": "छ", "more": "धेरै", "less": "कम"},
        "temperature": {"noun": "तापक्रम", "poss": "को", "poss_obl": "को", "whose": "कसको", "howmuch": "कति",
                        "unit": "डिग्री सेल्सियस", "copula": "छ", "more": "धेरै", "less": "कम"},
    },

    "templates": {
        "fact": [
            "{E}{poss} {noun} {v} {unit} {copula}।",
            "दिइएको जानकारी अनुसार {E}{poss} {noun} {v} {unit} {copula}।",
            "यो थाहा छ कि {E}{poss} {noun} {v} {unit} {copula}।",
            "{E}{poss} {noun} ठीक {v} {unit} {copula}।",
        ],
        "rel_fact": [
            "{A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} {copula}।",
            "दिइएको जानकारी अनुसार {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} {copula}।",
            "यो थाहा छ कि {A}{poss} {noun}, {B}{poss_obl} {noun} को तुलनामा {cmp} {copula}।",
        ],
        "q_which_of_two": [
            "{A} र {B} मध्ये {whose} {noun} {cmp} छ?",
            "{A} र {B} को तुलना गर्नुहोस्: {whose} {noun} {cmp} छ?",
            "बताउनुहोस् कि {A} र {B} मध्ये {whose} {noun} {cmp} छ?",
            "माथिको जानकारीको आधारमा {A} र {B} मध्ये {whose} {noun} {cmp} छ?",
        ],
        "q_extreme": [
            "यीमध्ये {whose} {noun} सबैभन्दा {cmp} छ?",
            "माथि उल्लेख भएकामध्ये {whose} {noun} सबैभन्दा {cmp} छ?",
            "बताउनुहोस् कि यी सबैमा {whose} {noun} सबैभन्दा {cmp} छ?",
            "दिइएको जानकारीको आधारमा {whose} {noun} सबैभन्दा {cmp} छ?",
        ],
        "q_equality": [
            "के {A} र {B}{poss} {noun} बराबर छ?",
            "के {A}{poss} {noun} र {B}{poss} {noun} समान छ?",
            "बताउनुहोस् कि {A} र {B}{poss} {noun} एकै छ कि छैन?",
            "के यी दुई, {A} र {B}{poss} {noun} बराबर छ?",
        ],
        "q_difference": [
            "{A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {howmuch} {cmp} छ?",
            "बताउनुहोस् कि {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {howmuch} {cmp} छ?",
            "{A} र {B}{poss} {noun} को अन्तर कति छ?",
            "माथिको जानकारी अनुसार {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {howmuch} {cmp} छ?",
        ],
        "q_ordering": [
            "यी सबैलाई {noun} अनुसार {cmp_from} बाट {cmp_to} क्रममा लेख्नुहोस्।",
            "माथिका सबैलाई {noun}{poss} आधारमा {cmp_from} देखि {cmp_to} क्रममा मिलाउनुहोस्।",
            "यिनलाई {noun} अनुसार {cmp_from} देखि {cmp_to} क्रममा बताउनुहोस्।",
            "दिइएको जानकारी अनुसार यी सबैलाई {noun} अनुसार {cmp_from} बाट {cmp_to} क्रममा राख्नुहोस्।",
        ],
        "q_middle": [
            "यी तीनमध्ये {whose} {noun} न सबैभन्दा धेरै छ न सबैभन्दा कम?",
            "{noun}{poss} आधारमा यी तीनमध्ये बीचमा कुन पर्छ?",
            "यी तीनमध्ये {whose} {noun} बीचमा छ?",
            "दिइएको जानकारी अनुसार {noun}{poss} क्रममा बीचमा कुन छ?",
        ],
        "q_verify": [
            "के यो सही छ कि {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} छ?",
            "बताउनुहोस् कि यो कथन सही छ कि छैन: {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} छ।",
            "के {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} छ?",
            "माथिको जानकारीको आधारमा बताउनुहोस् कि {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} छ कि छैन?",
        ],

        "fact_step": [
            "{E}{poss} {noun} = {v} {unit}।",
            "{E}: {noun} {v} {unit}।",
        ],
        "rel_step": [
            "{A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} छ।",
            "तुलना गर्दा {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} छ।",
        ],
        "deduce_step": [
            "{A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp} छ र {B}{poss} {noun}, {C}{poss_obl} {noun} भन्दा {cmp} छ, त्यसैले {A}{poss} {noun}, {C}{poss_obl} {noun} भन्दा {cmp} छ।",
            "{A} देखि {B} र {B} देखि {C} को श्रृंखला जोड्दा {A}{poss} {noun}, {C}{poss_obl} {noun} भन्दा {cmp} छ।",
        ],
        "flip_step": [
            "{A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp_from} छ, अर्थात् {B}{poss} {noun}, {A}{poss_obl} {noun} भन्दा {cmp_to} छ।",
            "उल्टो लेख्दा: {A}{poss} {noun}, {B}{poss_obl} {noun} भन्दा {cmp_from} छ, तसर्थ {B}{poss} {noun}, {A}{poss_obl} {noun} भन्दा {cmp_to} छ।",
        ],
        "diff_step": [
            "{vA} - {vB} = {d} {unit}।",
            "अन्तर निकाल्दा {vA} {unit} बाट {vB} {unit} घटाउँदा {d} {unit} हुन्छ।",
        ],
        "order_step": [
            "{noun}{poss} क्रम यस प्रकार छ: {ordered}।",
            "क्रममा मिलाउँदा: {ordered}।",
        ],
        "equal_step": [
            "{A} र {B}{poss} {noun} दुवै {v} {unit} छ, अर्थात् बराबर छ।",
            "दुवै{poss} {noun} समान छ: {v} {unit}।",
        ],
        "conclusion_step": [
            "त्यसैले उत्तर {answer} हो।",
            "यसकारण उत्तर {answer} हो।",
        ],
    },
}


SEED_PHRASEBANKS: dict[str, dict] = {
    "hindi": HINDI_SEED,
    "nepali": NEPALI_SEED,
}
