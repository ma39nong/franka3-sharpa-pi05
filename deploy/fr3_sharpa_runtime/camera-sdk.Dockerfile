# Build with the SHA-256 verified official SDK archive as described in README.md.
FROM franka-upper-body-teleop:latest
COPY sdk/ /opt/openpi-orbbec-sdk/
COPY camera_sdk_entrypoint.sh /opt/openpi-orbbec-sdk/entrypoint.sh
ENTRYPOINT ["bash", "/opt/openpi-orbbec-sdk/entrypoint.sh"]
