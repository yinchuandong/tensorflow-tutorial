import spacy

# %%
nlp = spacy.load('en_core_web_md')
# %%
nlp.to_disk('tmp-spacy')

nlp.vocab.vectors.from_glove('tmp-spacy-vectors/')
# %%
# %%
