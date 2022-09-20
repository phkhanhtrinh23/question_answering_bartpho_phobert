import json
import datasets

logger = datasets.logging.get_logger(__name__)

class ViQuADTestConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ViQuADTestConfig, self).__init__(**kwargs)


class ViQuADTest(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ViQuADTestConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="UIT-ViQuAD2.0",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": "data/test.json"}),
        ]

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                        }
                        key += 1