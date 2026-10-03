# Security Notice

This code is a research prototype for a controlled experimental environment.

- The Flask server has no authentication or authorization layer.
- Client-server communication uses plain HTTP and does not provide transport encryption.
- Tensor payloads are deserialized with PyTorch. Only load data from a trusted client and a trusted source.
- Do not expose the server directly to the public Internet.
- Run the software in an isolated environment and restrict network access to the intended experiment hosts.
- Review and update dependencies before using the code outside the documented research setting.

No production security guarantees are provided.
