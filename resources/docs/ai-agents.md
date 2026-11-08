# AI agents

ALIEN contains a built-in **MCP server**. MCP, the Model Context Protocol, is an open standard that lets AI agents use external programs. Once an agent is connected, you can describe in plain language what you want, and the agent operates ALIEN for you. It can create and load worlds, draw matter, design genomes and place creatures, change simulation parameters, run experiments, read statistics and look at the result through screenshots.

This is an easy way to get started with ALIEN. You do not need to know where each setting is, because the agent finds it. At the same time, the command log shows every step, so you can learn how things work.

## Connecting an agent

1. Open **Windows > MCP server** (Alt+6). The button with the agent icon in the edit mode tool bar also opens this window.
2. Click **Start server**. The status changes to *Running*. ALIEN remembers this and starts the server automatically next time.
3. In your AI agent, add an MCP server of type *HTTP*, which some agents call *Streamable HTTP*. Use the URL shown in the window. By default it is `http://127.0.0.1:8765/mcp`. The **Copy** button copies it to the clipboard.
4. Ask the agent to do something in ALIEN.

Any agent or chat application that supports MCP servers of type HTTP works, regardless of the AI provider, as long as it connects to ALIEN from your own computer. With Claude Code, for example, a single command adds ALIEN:

```
claude mcp add --transport http alien http://127.0.0.1:8765/mcp
```

> **Note:** Chat applications that connect to MCP servers through their provider's servers on the internet cannot reach ALIEN, even when you use them on this computer. If port 8765 is already in use, choose another one with the **Settings** button of the **MCP server** window and update the URL in your agent.

The **Command log** in the **MCP server** window lists the commands the agent sends, together with their results. This makes it easy to follow what the agent is doing. The log keeps the latest 1000 commands. Hover over a command or a result to see more of it in a tooltip.

## Example prompts

You can phrase your requests as you would to a knowledgeable assistant. The following prompts show what is possible. The results are best when you describe precisely what you want to see.

### Getting to know ALIEN

- "What is loaded in ALIEN right now? Describe the world and the creatures that live in it."
- "Find the most interesting creature, take a screenshot of it and explain how it is built."
- "Explain the simulation parameters of this world in simple terms. Which ones matter most?"
- "Which lineages are the largest and how do they differ?"

### Building worlds

- "Create a new world of 1000 x 600 with a rocky ground and a lake of glowing blue liquid."
- "Add gravity that pulls everything downwards, but only in the lower half of the world."
- "Write the word ALIEN in large solid letters in the middle of the world and let them crumble."
- "Paint a slowly drifting nebula into the background."

### Designing creatures

- "Design a small creature that swims towards energy particles and place five of them."
- "Build a plant that grows from a seed on the ground and forms a flower at its tip."
- "Create a predator with sensors, muscles and attacker cells that hunts the other creatures."
- "Show me the genome of the selected creature and explain what each gene does."

### Evolution experiments

- "Set up an evolution experiment with simple self-replicating creatures, an external energy supply and moderate mutation rates. Run it for 50,000 time steps and report how the population develops."
- "My population dies out after a while. Check the energy balance and suggest better parameters."
- "Compare the genomes of the three largest lineages. What has evolved?"
- "Create a flashback, double the attack strength and tell me how the ecosystem reacts."
- "Halve the neural network mutation rates of the largest lineage and leave its other evolved rates untouched."
- "Limit the sensor range of all predators to 40. Show me first which genomes would change."

### Art and presentation

- "Create a mysterious deep-sea scene with glowing jellyfish-like creatures."
- "Grow fractal trees with flowers on a dark background and arrange them nicely."
- "Find a beautiful view of the current simulation and take a screenshot of it."

## Tips for working with agents

- **Save your work first.** The agent works on the simulation that is currently loaded in ALIEN. Save your simulation before you let the agent experiment, even if you want it to start a new world. When the agent creates, loads or downloads a simulation, ALIEN replaces the current one without asking.
- **Use flashbacks.** Ask the agent to create a flashback before larger changes. With **Load flashback** in the **Temporal control** window, you can then return to the saved world and time step. Changes to the simulation parameters are not undone, apart from the **External energy amount** and the positions of moving layers and radiation sources. There is only one flashback, and a new one replaces the previous one. ALIEN also replaces the flashback with the resulting state whenever the agent creates, loads, downloads or resizes a world, or applies mass operations to the whole world.
- **Work in steps.** Large tasks succeed more reliably when you split them into steps and check the result in between. Screenshots help the agent see what happened.
- **Let it read the documentation.** Through the MCP server, the agent can read this documentation, including a detailed reference of all cell types and parameters. If the agent seems unsure, ask it to read the relevant chapter first.
- **Avoid overloading ALIEN.** Every command runs inside ALIEN. Many requests in quick succession on a large world can make ALIEN respond slowly. If that happens, ask the agent to slow down or to work on a smaller world.
- **Stay in control.** Through ALIEN, an agent can save and load files, and download from the ALIEN server. If you are logged in, it can also upload the current simulation to **My uploads** on the server, visible only to you. Use the command log to check what it does, and stop the MCP server when you do not need it.

## Available tools

The agent sees a description of every tool, so you do not need to know them. The list shows what an agent can and cannot do.

| Group | Tools |
| --- | --- |
| Temporal control | `get_time_info`, `run_simulation`, `pause_simulation`, `step_forward`, `step_backward`, `create_flashback`, `load_flashback`, `set_speed_limit` |
| Simulations and files | `create_simulation`, `resize_world`, `save_simulation`, `load_simulation`, `list_network_resources`, `download_network_resource`, `upload_simulation` |
| World editing | `create_object`, `create_rectangle`, `create_hexagon`, `create_disc`, `draw_freehand`, `create_line`, `create_curve`, `create_polygon`, `create_pattern_from_image`, `select_area`, `select_at`, `clear_selection`, `get_selection`, `delete_selection`, `fix_selection`, `color_selection`, `set_selection_sticky`, `move_selection`, `rotate_selection`, `relax_selection`, `glue_selection`, `connect_selection_to_surroundings`, `cut_connections`, `set_selection_velocity`, `set_selection_angular_velocity`, `uniform_selection_velocities`, `copy_selection`, `paste_selection`, `apply_force`, `multiply_selection_grid`, `multiply_selection_random`, `undo_multiplication`, `apply_mass_operations` |
| Genomes | `get_genome`, `save_genome`, `create_seed`, `inject_genome`, `mass_edit_genomes`, `summarize_genomes` |
| Inspection and view | `get_simulation_info`, `get_statistics`, `set_view`, `take_screenshot`, `find_objects`, `inspect_objects`, `change_object`, `get_json_format` |
| Simulation parameters | `list_parameter_groups`, `get_parameters`, `set_parameters`, `enable_expert_settings`, `reset_parameters`, `list_locations`, `add_layer`, `add_radiation_source`, `clone_location`, `delete_location`, `move_location`, `load_parameters`, `save_parameters` |
| Documentation | `read_documentation` |
