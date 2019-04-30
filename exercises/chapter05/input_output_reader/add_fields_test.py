def test():
    assert len(instances) == 2, "You didn't get two instances"
    expected_fields = {'text', 'title', 'stars', 'aspect', 'sentiment'}
    assert instances[0].fields.keys() == expected_fields, "You don't have the right fields in your Instance"
    assert instances[0]['sentiment'] == 'negative', "You didn't read the fields correctly"
    assert instances[0]['aspect'] == 'tutorials', "You didn't read the fields correctly"
    assert instances[1]['sentiment'] == 'positive', "You didn't read the fields correctly"
    assert instances[1]['aspect'] == 'library', "You didn't read the fields correctly"
    __msg__.good("Well done!")
