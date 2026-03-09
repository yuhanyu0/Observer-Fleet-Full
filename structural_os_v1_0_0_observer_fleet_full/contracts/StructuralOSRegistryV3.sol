// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract StructuralOSRegistryV3 {
    bytes32 public lastStatusHash;
    bytes32 public lastEventsHash;
    bytes32 public lastSchemaHash;
    string public lastURI;
    address public owner;

    event Published(
        uint32 indexed yyyymmdd,
        bytes32 statusHash,
        bytes32 eventsHash,
        bytes32 schemaHash,
        string uri,
        address publisher,
        uint256 timestamp
    );

    constructor() {
        owner = msg.sender;
    }

    function publish(
        uint32 yyyymmdd,
        bytes32 statusHash,
        bytes32 eventsHash,
        bytes32 schemaHash,
        string calldata uri
    ) external {
        require(msg.sender == owner, "not owner");
        lastStatusHash = statusHash;
        lastEventsHash = eventsHash;
        lastSchemaHash = schemaHash;
        lastURI = uri;
        emit Published(yyyymmdd, statusHash, eventsHash, schemaHash, uri, msg.sender, block.timestamp);
    }
}
