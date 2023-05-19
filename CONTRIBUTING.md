# Contributing to FedEra

Thank you for investing your time in contributing to our project. This document explains how to contribute to FedERA. 

We accept various contributions from documentation improvement and bug fixing to major features proposals.

To propose bugs, new features, or other code improvements:

1. Familiarize Yourself with the Project:
Read the documentation, explore the codebase, and understand the overall architecture and goals of FedERA. Familiarize yourself with any existing guidelines and contributing documentation.

2. Choose a Contribution Type:
Determine the type of contribution you would like to make. It could be fixing a bug, implementing a new feature, improving documentation, optimizing performance, or any other valuable contribution.

3. Create an Issue (Optional):
If you're working on a bug fix or a specific feature, consider creating an issue in the issue tracker. Describe the problem or feature request clearly, provide context and any relevant information, and assign appropriate labels or tags. This allows others to provide feedback and track the progress of your contribution.

4. Fork the Repository:
Fork the main repository to your own GitHub account. This creates a copy of the project that you can freely modify.

5. Create a New Branch:
Create a new branch in your forked repository. Give it a descriptive and meaningful name that reflects the purpose of your contribution. This helps keep the codebase organized and makes it easier for others to understand your changes.

6. Make Your Changes:
Make the necessary changes to the codebase based on the contribution type you chose. Follow the coding standards and guidelines established by the project. Write clean, well-documented code and consider adding unit tests for new functionality or modifications.

7. Commit Your Changes:
Commit your changes to the branch you created. Provide a clear and concise commit message that describes the purpose of the changes. It's good practice to keep each commit focused on a single logical change.

8. Push Your Changes:
Push your branch with the committed changes to your forked repository on GitHub. This makes your changes accessible for others to review.

9. Submit a Pull Request:
Open a pull request (PR) from your branch in the forked repository to the main repository. Provide a detailed description of the changes you made, including the problem you solved or the feature you implemented. Mention any relevant issue numbers or related discussions.

10. Engage in Code Review:
Participate in the code review process by responding to feedback and addressing any requested changes. Collaborate with the project maintainers and other contributors to refine your contribution. Be open to constructive criticism and embrace the opportunity to learn and improve your work.

11. Merge and Close the Pull Request:
Once your changes have been reviewed and approved by the project maintainers, they will merge your branch into the main repository. At this point, your contribution becomes part of the project. The corresponding issue (if any) can be closed, and your work will be attributed to you.


### Formatting of Pull Requests

FedERA follows standard recommendations of PR formatting. Please find more details [here](https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/).

### Continuous Integration and Continuous Development

FedERA uses GitHub actions to perform all functional and unit tests. Before your contribution can be merged make sure that all your tests are passing. 
For more information of what fails you can click on the “details” link near the pipeline that failed.

### Writing the tests

The FedERA team recommend including tests for all new features contributions. Test can be found in the “Tests” directory. 
The [Tests/ folder](https://github.com/akshatbhandari15/FedERA/tree/main/test) contains unit tests and the [Tests/unittest folder](https://github.com/akshatbhandari15/FedERA/tree/main/test/unittest) contains end-to-end and functional tests.

### License

FedERA is licensed under the terms in [Apache 2.0 license](https://github.com/akshatbhandari15/FedERA/blob/main/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch.
