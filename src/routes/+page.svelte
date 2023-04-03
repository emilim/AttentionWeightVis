<script lang="ts">
   import { onMount } from "svelte";
   import * as d3 from "d3-force";
   import * as neo4j from "neo4j-driver";
   import ForceGraph3D from "3d-force-graph";
   import ForceGraph2D from "force-graph";

   import {
      PUBLIC_NEO4J_URI,
      PUBLIC_NEO4J_USERNAME,
      PUBLIC_NEO4J_PASSWORD,
   } from "$env/static/public";


   onMount(() => {
      const driver = neo4j.driver(
         PUBLIC_NEO4J_URI,
         neo4j.auth.basic(PUBLIC_NEO4J_USERNAME, PUBLIC_NEO4J_PASSWORD)
      );
      const session = driver.session({ database: 'neo4j' });

      //delete all nodes and relationships
      // 'MATCH (n) DETACH DELETE n'

      // get all nodes and relationships
      // 'MATCH (n) OPTIONAL MATCH (n)-[r]-() RETURN n, r;'

      function toColor(num: number) {
         num = Math.floor(num * 255)
         num >>>= 0;
         var g = num & 0xFF,
            r = (num & 0xFF00) >>> 8,
            b = (num & 0xFF0000) >>> 16
         //return "rgba(" + [r, g, b, a].join(",") + ")";
         return "rgb(" + [r, g, b].join(",") + ")";;
      }

      session
         .run(
            `
            MATCH (s:State)-[t:Transition]->(s2:State)
            RETURN s.id AS source, s2.id AS target, t.prob AS value
         `
         )
         .then((result) => {
            const nodes = [] as any[];
            const links = [] as any[];
            const existingNodeIds = new Set<string>();

            result.records.forEach((record) => {
               const sourceId = String(record.get("source"));
               const targetId = String(record.get("target"));
               const value = Number(record.get("value"));

               if (!existingNodeIds.has(sourceId)) {
                  existingNodeIds.add(sourceId);
                  nodes.push({ id: sourceId, name: sourceId });
               }
               if (!existingNodeIds.has(targetId)) {
                  existingNodeIds.add(targetId);
                  nodes.push({ id: targetId, name: targetId });
               }
               //nodes.push({ id: sourceId });
               //nodes.push({ id: targetId });
               links.push({ source: sourceId, target: targetId, value, color: toColor(value*255), name: value.toFixed(2) });
            });

            const Graph = ForceGraph3D()(
               document.getElementById("graph-container") as HTMLElement
            )
               .graphData({ nodes, links })
               .nodeAutoColorBy("id")
               .d3Force("charge", d3.forceManyBody().strength(-50))
               .d3Force("link", d3.forceLink().distance(20))
               .d3Force("center", d3.forceCenter())
               .cooldownTicks(200);
         })
         .catch((error) => {
            console.error(error);
         })
         .finally(() => {
            session.close();
            driver.close();
         });
   });
</script>

<div id="graph-container" />
<div id="node-info" />

<style>
   #graph-container {
      width: 100%;
      height: 600px;
   }
</style>
