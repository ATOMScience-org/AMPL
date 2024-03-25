# Table of Contents

<!-- toc -->
- [Contributing to AMPL](#contributing-to-ampl)
  - [Getting Started](#getting-started)
  - [Pull Request Process](#pull-request-process)
  - [Coding Conventions](#coding-conventions)
  - [Documentation Conventions](#document-conventions)
- [The Agreement](#the-agreement)
- [AMPL Technical Steering Committee](#ampl-technical-steering-committee)
<!-- tocstop -->

## Contributing to AMPL

We actively encourage community contributions to AMPL. The first
place to start getting involved is
[the tutorials](https://github.com/ATOMScience-org/AMPL/tree/master/atomsci/ddm/examples).

Afterwards, we encourage contributors to give a shot to improving our documentation.
While we take effort to provide good docs, there's plenty of room
for improvement. All docs are hosted on [Github](https://github.com/ATOMScience-org/AMPL) in `README.md` files and on [ReadtheDocs](https://ampl.readthedocs.io/en/latest/).

Once you've got a sense of how AMPL works, we encourage the use
of Github issues to discuss more complex changes, raise requests for
new features or propose changes to the global architecture of AMPL.
Once consensus is reached on the issue, please submit a PR with proposed
modifications. All contributed code to AMPL will be reviewed by a member
of the AMPL team, so please make sure your code style and documentation
style match our guidelines!

### Getting Started

To develop AMPL on your machine, we recommend using pip for managing
packages. If you want to manage multiple builds of AMPL, you can make use of
[python virtual environments](https://docs.python.org/3/library/venv.html)
to maintain separate Python package environments, each of which can be tied
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

Every contribution must be a pull request and must have adequate time for
review by other committers.

A member of the AMPL team will review the pull request.
The default path of every contribution should be to merge to the current dev branch, usually named as upcoming version tag, such as branch 1.6.0. The discussion,
review, and merge process should be designed as corrections or improvements that move the
contribution into the path to merge. Once there are no more corrections,
(dissent) changes should merge without further process. 

### Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

*Thank you [OpenFL](https://github.com/securefederatedai/openfl/blob/develop/CONTRIBUTING.md) for this section*

### Coding Conventions

AMPL uses these tools for keeping our codes healthy.

- [pytest](https://docs.pytest.org/en/6.2.x/index.html) (unit and integrative testing)

Before making a PR, please check your codes using them and include tests, both unit and integrative, 
for testing your contribution. PRs will not be accepted unless they 
come with tests.

### Document Conventions

AMPL uses [Sphinx](https://www.sphinx-doc.org/en/master/) to build
[the documentation](https://ampl.readthedocs.io/en/latest/).
The document is automatically built by
[Numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide)
in source codes.
For any changes or modification to source code in a PR, please don't forget to add or modify Numpy style docstrings.

## The Agreement
AMPL is licensed under the terms in MIT License. By contributing to the project, 
you agree to the license and copyright terms therein and release your contribution under these terms.

## AMPL Technical Steering Committee

The Technical Steering Committee oversees all top-level administration of AMPL.

The TSC exercises autonomy in setting up and maintaining procedures, policies,
and management and administrative structures as it deems appropriate for the
maintenance and operation of these projects and resources.

Included in the responsibilities of the TSC are:

* Managing code and documentation creation and changes for the listed projects and resources
* Performing code reviews on incoming pull requests and merging suitable code changes
* Setting and maintaining standards covering contributions of code, documentation and other materials
* Managing code and binary releases: types, schedules, frequency, delivery mechanisms
* Making decisions regarding dependencies of AMPL, including what those
dependencies are and how they are bundled with source code and releases
* Creating new repositories and projects under the @ATOMscience GitHub organization as required
* Setting overall technical direction for the AMPL project, including 
high-level goals and low-level specifics regarding features and functionality
* Setting and maintaining appropriate standards for community discourse via the various
mediums under TSC control 

Members of the TSC will meet regularly (over phone or video conferencing)
to coordinate efforts.
The current members of the TSC are (alphabetically)
* Jonathan Allen
* Sean Black 
* Stewart He
* Jessica Mauvais
* Kevin McLoughlin
* Amanda Paulson
