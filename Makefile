.DEFAULT_GOAL := help
.PHONY: git-config help

git-config:  ## Common Git configuration
	git config blame.ignoreRevsFile .git-blame-ignore-revs

help:  ## Display this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
