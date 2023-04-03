import { PUBLIC_NEO4J_URI, PUBLIC_NEO4J_USERNAME, PUBLIC_NEO4J_PASSWORD } from "$env/static/public";

import neo4j from 'neo4j-driver';

const driver = neo4j.driver(PUBLIC_NEO4J_URI, neo4j.auth.basic(PUBLIC_NEO4J_USERNAME, PUBLIC_NEO4J_PASSWORD));

export default driver;