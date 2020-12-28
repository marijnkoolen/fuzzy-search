# auction advertisement structure
# broker names
# Makelaars
# tot {geo}
# zal|zullen op
# {weekday} # optional
# den {day_month},
# 's {day_part} ten {hour} uuren precies # optional
# in|tot {geo}
# in de {street} #optional
# in 't {building} # optional
# verkopen
# 't Amft. in de Nes in de Brakke Grond


auction_template = template = {
    "type": "group",
    "ordered": True,
    "label": "auction_template",
    "elements": [
        {
            "type": "group",
            "label": "broker_grouping",
            "ordered": False,
            "elements": [
                {"type": "label", "label": "broker_name", "cardinality": "multi", "variable": True},
                {"type": "label", "label": "broker_term", "required": True}
            ]
        },
        {
            "type": "group",
            "label": "auction_grouping",
            "ordered": True,
            "elements": [
                {"type": "label", "label": "auction_date"},
                {"type": "label", "label": "auction_time"},
                {"type": "label", "label": "auction_location"},
                {"type": "label", "label": "auction_sale"}
            ]
        },
        {
            "type": "group",
            "label": "product_grouping",
            "ordered": False,
            "required": True,
            "elements": [
                {"type": "label", "label": "product_type", "cardinality": "multi", "required": True},
                {"type": "label", "label": "product_unit", "cardinality": "multi"},
                {"type": "label", "label": "product_specifier", "cardinality": "multi"}
            ]
        }
    ]
}


auction_broker_phrases = [
    {
        "phrase": "Makelaars",
        "label": "broker_term",
        "variants": [
            "Makelaer",
            "Makelaers",
            "Maakelaar",
            "Maakelaars",
            "Makelaar",
        ],
        "distractors": [
            "Martelaars",
            "Zadelaars",
            "Martelaaren",
            "Kandelaars",
            "Metselaars"
        ]
    }
]


auction_location_phrases = [
    {
        "phrase": "te Amsterdam",
        "label": ["auction_location", "auction_amsterdam"]
    },
    {
        "phrase": "in de Nes in de Brakke Grond",
        "label": ["auction_location", "auction_amsterdam"]
    }
]


auction_date_phrases = [
    {
        "phrase": "zullen op",
        "label": "auction_sale",
        "variants": [
            "zullen",
            "zullen op",
            "zal op",
            "sullen op",
            "sal op"
        ]
    },
    {
        "phrase": "verkopen",
        "label": "auction_sale",
        "variants": [
            "verkoopen",
            "vercopen",
            "vercoopen"
        ],
        "distractor": [
            "verkoper",
            "verkooper"
        ]
    },
    {
        "phrase": "presenteren",
        "label": "auction_sale",
    },
    {
        "phrase": "meestbiedende",
        "label": "auction_sale",
    },
    {
        "phrase": "Maandag",
        "label": [
            "auction_weekday",
            "auction_date"
        ],
        "variants": [
            "Maendag"
        ]
    },
    {
        "phrase": "Dinsdag",
        "label": [
            "auction_weekday",
            "auction_date"
        ]
    },
    {
        "phrase": "Woensdag",
        "label": [
            "auction_weekday",
            "auction_date"
        ]
    },
    {
        "phrase": "Donderdag",
        "label": [
            "auction_weekday",
            "auction_date"
        ]
    },
    {
        "phrase": "Vrijdag",
        "label": [
            "auction_weekday",
            "auction_date"
        ],
        "variants": [
            "Vrydag"
        ]
    },
    {
        "phrase": "Zaterdag",
        "label": [
            "auction_weekday",
            "auction_date"
        ],
        "variants": [
            "Saturdag"
        ]
    },
    {
        "phrase": "Zondag",
        "label": [
            "auction_weekday",
            "auction_date"
        ]
    },
    {
        "phrase": "",
        "label": [
            "auction_weekday",
            "auction_date"
        ]
    },
    {
        "phrase": "Januari",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "distractor": [
            "Februari"
        ]
    },
    {
        "phrase": "Februari",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "Feb"
        ],
        "distractor": [
            "Januari"
        ]
    },
    {
        "phrase": "Maart",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "distractor": [
            "Mei"
        ]
    },
    {
        "phrase": "April",
        "label": [
            "auction_month",
            "auction_date"
        ]
    },
    {
        "phrase": "Mei",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "Mey",
            "May"
        ],
        "distractor": [
            "Maart"
        ]
    },
    {
        "phrase": "Juni",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "Juny"
        ],
        "distractor": [
            "Juli",
            "Junior"
        ]
    },
    {
        "phrase": "Juli",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "July"
        ],
        "distractor": [
            "Juni"
        ]
    },
    {
        "phrase": "Augustus",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "Auguftus",
            "Auguftuf",
            "Aug"
        ]
    },
    {
        "phrase": "September",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "Sept"
        ],
        "distractor": [
            "Oktober",
            "November",
            "December"
        ]
    },
    {
        "phrase": "Oktober",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "Okt"
        ],
        "distractor": [
            "September",
            "November",
            "December"
        ]
    },
    {
        "phrase": "November",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "Nov"
        ],
        "distractor": [
            "Oktober",
            "September",
            "December"
        ]
    },
    {
        "phrase": "December",
        "label": [
            "auction_month",
            "auction_date"
        ],
        "variants": [
            "Dec"
        ],
        "distractor": [
            "Oktober",
            "November",
            "September"
        ]
    },
    {
        "phrase": "den eersten",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den tweeden",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den derden",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den vierden",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den vijfden",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den zesden",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den zevenden",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den achtsten",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den negenden",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den tienden",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    },
    {
        "phrase": "den 1",
        "label": [
            "auction_month_day",
            "auction_date"
        ]
    }
]

auction_product_phrases = [
    {
        "phrase": "Koffie",
        "label": [
            "product_type"
        ],
        "variants": [
            "Coffy",
            "Coffie",
            "Coffi",
            "Koffi"
        ]
    },
    {
        "phrase": "Koffiebonen",
        "label": [
            "product_type"
        ],
        "variants": [
            "Coffyboonen",
            "Koffiboonen",
            "Coffybonen",
            "Koffibonen",
            "Coffy-Boonen",
            "Coffy-Bonen"
        ]
    },
    {
        "phrase": "Thee",
        "label": [
            "product_type"
        ],
        "distractors": [
            "twee",
            "'t zee",
            "t vee",
            "otheek"
        ]
    },
    {
        "phrase": "Tabak",
        "label": [
            "product_type"
        ],
        "variants": [
            "TABAK"
        ]
    },
    {
        "phrase": "Surinaamse",
        "label": [
            "product_origin"
        ],
        "variants": [
            "Surinaemse",
            "Surinaemfe"
        ]
    },
    {
        "phrase": "Javaanse",
        "label": [
            "product_origin"
        ],
        "variants": [
            "Javaenfe",
            "Javaanfe"
        ]
    },
    {
        "phrase": "Bourbonse",
        "label": [
            "product_origin"
        ],
        "variants": [
            "Bourbonfe"
        ]
    },
    {
        "phrase": "Havana",
        "label": [
            "product_origin"
        ]
    },
    {
        "phrase": "Londonse",
        "label": [
            "product_origin"
        ],
        "variants": [
            "Londonfe"
        ]
    },
    {
        "phrase": "Brazil",
        "label": [
            "product_origin"
        ]
    },
    {
        "phrase": "Vaten",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Vaatjes"
        ]
    },
    {
        "phrase": "Balen",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Baalen",
            "Baelen",
            "Baaltjes",
            "Baeltjes"
        ]
    },
    {
        "phrase": "Kisten",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Kistjes",
            "Kisjes",
            "Kist"
        ]
    },
    {
        "phrase": "Fusten",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Fust",
            "Fuft",
            "Fuften"
        ]
    },
    {
        "phrase": "Bussen",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Bus",
            "Buf",
            "Buffen"
        ],
        "distractors": [
            "tuffen",
            "tussen"
        ]
    },
    {
        "phrase": "Flessen",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Fles",
            "Flesje",
            "Flesjes",
            "Flef",
            "Felfje",
            "Flefjes",
            "Fleffen"
        ]
    },
    {
        "phrase": "Dozen",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Doozen",
            "Doos",
            "Doof"
        ],
        "distractors": [
            "Dezen",
            "Doelen",
            "Roos",
            "Rozen",
            "Hoos"
        ]
    },
    {
        "phrase": "Scheepslading",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Scheeps-lading",
            "Scheepsladingen",
            "Scheeps-ladingen"
        ]
    },
    {
        "phrase": "Rollen",
        "label": [
            "product_unit"
        ]
    },
    {
        "phrase": "Partij",
        "label": [
            "product_unit"
        ],
        "variants": [
            "Party"
        ]
    },
    {
        "phrase": "Groene",
        "label": [
            "product_specifier",
            "Thee"
        ],
        "distractors": [
            "Greene"
        ]
    },
    {
        "phrase": "Keyzers",
        "label": [
            "product_specifier",
            "Thee"
        ]
    },
    {
        "phrase": "Boey",
        "label": [
            "product_specifier",
            "Thee"
        ],
        "distractors": [
            "Baey",
            "Boel",
            "Boete",
            "Boek"
        ]
    },
    {
        "phrase": "Congoe",
        "label": [
            "product_specifier",
            "Thee"
        ],
        "variants": [
            "Congo",
            "Cungo"
        ]
    },
    {
        "phrase": "Pekoe",
        "label": [
            "product_specifier",
            "Thee"
        ],
        "variants": [
            "Pecco",
            "Pecoe"
        ]
    },
    {
        "phrase": "Soachon",
        "label": [
            "product_specifier",
            "Thee"
        ],
        "variants": [
            "Choachon"
        ]
    },
    {
        "phrase": "Fyne",
        "label": [
            "product_specifier",
            "Thee"
        ]
    },
    {
        "phrase": "Suprafyne",
        "label": [
            "product_specifier",
            "Thee"
        ]
    },
    {
        "phrase": "Heyzan",
        "label": [
            "product_specifier",
            "Thee"
        ],
        "variants": [
            "Heyssan",
            "Heyffan"
        ]
    },
    {
        "phrase": "Hombouc",
        "label": [
            "product_specifier",
            "Thee"
        ]
    },
    {
        "phrase": "Souchou",
        "label": [
            "product_specifier",
            "Thee"
        ]
    },
    {
        "phrase": "Varinas",
        "label": [
            "product_specifier",
            "Tabak"
        ]
    },
    {
        "phrase": "Virginy",
        "label": [
            "product_specifier",
            "Tabak"
        ],
        "variants": [
            "Verginy",
            "Virginie"
        ]
    },
    {
        "phrase": "Sweetsented",
        "label": [
            "product_specifier",
            "Tabak"
        ],
        "variants": [
            "Swietsented",
            "Zweetsented",
            "Zwietsented"
        ]
    },
    {
        "phrase": "Bladeren",
        "label": [
            "product_specifier",
            "Tabak"
        ],
        "distractors": [
            "Bladen"
        ]
    },
    {
        "phrase": "Bladen",
        "label": [
            "product_specifier",
            "Tabak"
        ],
        "distractors": [
            "Blanken",
            "Bladeren"
        ]
    },
    {
        "phrase": "Baey",
        "label": [
            "product_specifier",
            "Tabak"
        ],
        "distractors": [
            "Boey"
        ]
    },
    {
        "phrase": "Snuif",
        "label": [
            "product_specifier",
            "Tabak"
        ]
    },
    {
        "phrase": "Surinaamse",
        "label": [
            "product_specifier",
            "Koffie"
        ],
        "variants": [
            "Surinaemse",
            "Surinaemfe"
        ]
    },
    {
        "phrase": "Javaanse",
        "label": [
            "product_specifier",
            "Koffie"
        ],
        "variants": [
            "Javaenfe",
            "Javaanfe"
        ]
    },
    {
        "phrase": "Havana",
        "label": [
            "product_specifier",
            "Koffie"
        ]
    },
    {
        "phrase": "Bourbonse",
        "label": [
            "product_specifier",
            "Koffie"
        ],
        "variants": [
            "Bourbonfe"
        ]
    },
    {
        "phrase": "Mochase",
        "label": [
            "product_specifier",
            "Koffie"
        ]
    }
]


auction_phrases = auction_broker_phrases + auction_date_phrases + auction_location_phrases + auction_product_phrases

# to do
day_parts = ["ochtends", "voormiddags", "namiddags", "middags", "avonds"]
hours = ["een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen", "tien", "elf", "twaalf"]
start_time = ["ten 1 uur", "ten 4 uuren", "ten 1 uur precies", "ten 4 uuren precies", ]

auction_texts = [
    """ooi woont. ~ Antony van Ruynen,Nicolaes de Sterke, Jacobus van der Hoeven en Fskncois Van der Leven,Makelaer»,fullen Maendag.den jq Maerr, te Rotterdam in dc Hbrberg de fwart/Lceu in dc Wijnftraet vtrkopen een Scheeps-Lading puykx puyk Virgini Bladeren Bay-Taback, nu eerft direft uyt de Virginien gearriveert: De Taback is léggende in de Boompjes in 't Packhuys van de Heer Jaccb Sencerf. Tacob Magnus de jon""",
    """e Taback is léggende in de Boompjes in 't Packhuys van de Heer Jaccb Sencerf. Tacob Magnus de jonge,Makelaer,lalWoensdag,den Maert, t'Amiterdam in dc Nes in de Bracke Gront verkopen een Party fijne oieuwc Modefe couleurde witte Lakenen en diverife Manufaduren, beftaende in Carfayen, Drogetten, fwarte Bowrattcn. Pietzen,Calaminken, Sareie dn Bois,Bayen,Frifaden,Sargies, Eftaminen, Krippen, Wolle Damaften,S""",
    """k aènftonts van alle Loteryenhaei Loten aengewefen : Allet voor een civUe Provifie. - ■ Simon Boom, Makelaen, fat Woensdag, den 21 September, tot Amfterdam pubhjck aen do Meelt biedenie verkopen een Party vaa j 6eeo Riemen FrarifTe Druck-Pasteren. , . • : • —• , T'Amfterdam by Jan Hak in deCoffy-Boom over de Beurs ts te bekomen :De Galante Ttjtverdrijver.vwer in in Rijm,foo wel morale.al» JjedendaegfeGele""",
    """g met en fonder Voeten,het glimp<«Ë|pei de en Rofematijn Blad, alle in Manden. 13. 'Adriaeo Doelman,Makelaer,fal Woensdag,den 6 April,te Rotterdam in de Herberg de fwarte Leeu In de Wijnftraet verkopen een etfflß'-n ordinaire curieufe Party van omtrent 70 a 80 Vaten ïrfle Dublin Talck, beftaende in Tabaka- en Suycker-Vaten en kleynder Soort (na u/t Zet geatr'veert cn noch gelofcht ftaende te werden) met C""",
    """kfe Deelen, en Rayynfe Maften. Hendrik Kofter, Michiel de Roode, Reynier Kofter en Jacob Abtabanel, Makelaers, pref enteren aen de meeftbiedendé'te veikopen een part-e Ixtraordinaris puyks puyk van onireut roooo pond Nieuwe Havana Tabaks-bladcn.en 238 ftuks Havana Huydcn.by Cavelingrn.le-'gende op he* Pakiioys van deWcft-Indifcbe Couipagnie.opßapenburg, op Maendag den 6 Miert 1702,'s ivonds ten 6 uuren pre"""
]

auction_tests = {
    "test1": {
        "matches_template": True,
        "match_sequence": [
            "product_type", # belongs to previous auction event
            "broker_term",
            "auction_date",
            "auction_sale",
            "product_type",
            "product_unit", # belongs to product type that is not found
            "product_unit", # belongs to product type that is not found
            "product_specifier", # belongs to product type that is not found
        ],
        "num_template_matches": 1,
        "template_matches": [[
            "broker_term",
            "auction_date",
            "auction_sale",
            "product_type",
            "product_unit",
            "product_unit",
            "product_specifier",
        ]]
    },
    "test2": {
        "matches_template": False,
        "match_sequence": [
            "product_type",
            "broker_term",
            "auction_date",
            "auction_sale",
            "product_unit",
            "product_unit",
            "product_specifier",
        ],
        "num_template_matches": 0,
        "template_matches": []
    },
    # auction date and sale after products, so not part of event
    "test3": {
        "matches_template": True,
        "match_sequence": [
            "product_type", # belongs to previous auction event
            "broker_term",
            "product_type",
            "product_unit",
            "product_unit",
            "product_specifier",
            "auction_date", # belongs to next auction event
            "auction_sale", # belongs to next auction event
        ],
        "num_template_matches": 1,
        "template_matches": [[
            "broker_term",
            "product_type",
            "product_unit",
            "product_unit",
            "product_specifier",
        ]]
    },
    # auction date and sale after products, so not part of event
    "test4": {
        "matches_template": True,
        "match_sequence": [
            "product_type", # belongs to previous auction event
            "broker_term",
            "product_type",
            "product_unit",
            "product_unit",
            "product_specifier",
            "broker_term",
            "auction_date", # belongs to next auction event
            "auction_sale", # belongs to next auction event
            "product_type",
        ],
        "num_template_matches": 2,
        "template_matches": [
            [
                "broker_term",
                "product_type",
                "product_unit",
                "product_unit",
                "product_specifier",
            ],
            [
                "broker_term",
                "auction_date",
                "auction_sale",
                "product_type"
            ]
        ]
    },
}
