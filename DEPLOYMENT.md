# Deploying QUEASARS

To deploy QUEASARS automatically to PyPI and Github Pages the following steps need to be taken:

## Deploying QUEASARS using tags (Recommended)

1. Make sure the project ``version`` field in the ``pyproject.toml`` under ``tools.poetry`` section is set to the version that should be deployed. It must follow the format ``[0-9].[0-9].[0-9]`` (e.g., 0.1.0).
2. Update the ``CHANGELOG.md`` accordingly.
3. Create and push a tag on the commit which should be published. The tag must be named according to the following format: ``v[0-9].[0-9].[0-9]`` (e.g., v0.1.0).
   1. The tag and pyproject.toml version numbers need to match (excluding the "v" in the tag name), otherwise the deployment action will fail.
   2. Only users with ``admin`` or ``maintain`` or ``edit repository rules`` permissions will be able to create the necessary tags due to them being protected.
4. QUEASARS is now automatically published to PyPI and GitHub Pages.


## Trigger the QUEASARS deployment manually (Not recommended)

The GitHub deployment action can also be triggered manually in the GitHub actions tab.
This may be useful, if only a partial deployment is needed (e.g., only the documentation should be deployed) from the latest commit of a chosen branch (no tag is created).
To do this, the following steps need to be taken:

1. Make sure the project ``version`` field in the ``pyproject.toml`` under ``tools.poetry`` section is set to the version that should be deployed. It must follow the format ``[0-9].[0-9].[0-9]`` (e.g., 0.1.0).
2. Update the ``CHANGELOG.md`` accordingly.
3. Go to the GitHub actions branch, choose the ``QUEASARS Deployment`` action.
4. Use the ``Run workflow`` button.
   1. Specify the version according to the following format: ``v[0-9].[0-9].[0-9]`` (e.g., v0.1.0).
   2. The specified version and pyproject.toml version numbers need to match (excluding the "v" in the specified version), otherwise the deployment action will fail.
   3. Choose whether deployment to PyPI or GitHub pages or both is needed.

