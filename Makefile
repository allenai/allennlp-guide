.PHONY : exercises-test
exercises-test :
	rm -rf _exercises_test/
	mkdir _exercises_test/
	python scripts/build_exercise_tests.py
	black _exercises_test/*.py
	mypy _exercises_test/
	flake8 _exercises_test/
