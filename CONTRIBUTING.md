# Table of Contents

<!-- toc -->
- [Contributing to AMPL](#contributing-to-ampl)
  - [Getting Started](#getting-started)
  - [Pull Request Process](#pull-request-process)
  - [Coding Conventions](#coding-conventions)
  - [Documentation Conventions](#documentation-conventions)
- [The Agreement](#the-agreement)
<!-- tocstop -->

## Contributing to AMPL

We actively encourage community contributions to AMPL. The first
place to start getting involved is
[the tutorials](https://github.com/ATOMScience-org/AMPL/tree/master/atomsci/ddm/examples).
Afterwards, we encourage contributors to give a shot to improving our documentation.
While we take effort to provide good docs, there's plenty of room
for improvement. All docs are hosted on Github, either in `README.md`
file, or in the `docs/` directory.

Once you've got a sense of how AMPL works, we encourage the use
of Github issues to discuss more complex changes, raise requests for
new features or propose changes to the global architecture of AMPL.
Once consensus is reached on the issue, please submit a PR with proposed
modifications. All contributed code to AMPL will be reviewed by a member
of the AMPL team, so please make sure your code style and documentation
style match our guidelines!

### Getting Started

To develop AMPL on your machine, we recommend using Anaconda for managing
packages. If you want to manage multiple builds of AMPL, you can make use of
[python virutal environments](https://docs.python.org/3/library/venv.html)
to maintain seperate Python package environments, each of which can be tied
to a specific build of AMPL. Here are some tips to get started:

1. Fork the [AMPL](https://github.com/ATOMScience-org/AMPL) repository
and clone the forked repository

```bash
git clone https://github.com/YOUR-USERNAME/AMPL.git
cd AMPL
```

&nbsp;&nbsp;&nbsp;&nbsp; 1.1. If you already have AMPL from source, update it by running
```bash
git fetch upstream
git rebase upstream/master
```

2. Set up a new environment for AMPL by following these [instructions](https://github.com/ATOMScience-org/AMPL#install).

Keep in mind, every contribution must pass the unit tests.

### Pull Request Process

Every contribution, must be a pull request and must have adequate time for
review by other committers.

A member of the AMPL team will review the pull request.
The default path of every contribution should be to merge. The discussion,
review, and merge process should be designed as corrections that move the
contribution into the path to merge. Once there are no more corrections,
(dissent) changes should merge without further process.

### Coding Conventions

AMPL uses these tools or styles for keeping our codes healthy.

- [pytest](https://docs.pytest.org/en/6.2.x/index.html) (unit testing)

Before making a PR, please check your codes using them.

### Document Conventions

AMPL uses [Sphinx](https://www.sphinx-doc.org/en/master/) to build
[the documentation](https://ampl.readthedocs.io/en/latest/).
The document is automatically built by
[Numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide)
in source codes.
For any changes or modification to source code in a PR, please don't forget to add or modify Numpy style docstrings.

## The Agreement

Contributor offers to license certain software (a “Contribution” or multiple
“Contributions”) to Lawrence Livermore National Security, LLC (LLNS), and LLNS agrees to accept said Contributions,
under the terms of the open source license [The MIT License](https://opensource.org/licenses/MIT)

The Contributor understands and agrees that LLNS shall have the
irrevocable and perpetual right to make and distribute copies of any Contribution, as
well as to create and distribute collective works and derivative works of any Contribution,
under [The MIT License](https://opensource.org/licenses/MIT).

LLNS understands and agrees that Contributor retains copyright in its Contributions.
Nothing in this Contributor Agreement shall be interpreted to prohibit Contributor
from licensing its Contributions under different terms from the
[The MIT License](https://opensource.org/licenses/MIT) or this Contributor Agreement.
