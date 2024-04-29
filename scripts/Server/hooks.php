<?php
    function sendDiscordMessage($payload) {
        if (strlen($payload) >= 0) {
            $ch = curl_init("[webhook URL]");
            curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-type: application/json'));
            curl_setopt($ch, CURLOPT_POST, 1);
            curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
            curl_setopt($ch, CURLOPT_FOLLOWLOCATION, 1);
            curl_setopt($ch, CURLOPT_HEADER, 0);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
            $response = curl_exec($ch);
            curl_close($ch);
            return $response;
        }
    }

    // functions for Discord messages
    function createAddResourceMessage($type, $simName, $userName, $simDesc, $width, $height, $particles) {
        if ($type == 0) {
            return createMessageForSimulation("New simulation added to the database", $simName, $userName, $simDesc, $width, $height, $particles);
        }
        else {
            return createMessageForGenome("New genome added to the database", $simName, $userName, $simDesc, $width, $height, $particles);
        }
    }

    function createUpdateResourceMessage($type, $simName, $userName, $simDesc, $width, $height, $particles) {
        if ($type == 0) {
            return createMessageForSimulation("Simulation updated in the database", $simName, $userName, $simDesc, $width, $height, $particles);
        }
        else {
            return createMessageForGenome("Genome updated in the database", $simName, $userName, $simDesc, $width, $height, $particles);
        }
    }

    function createMessageForSimulation($message, $simName, $userName, $simDesc, $width, $height, $particles) {
        $particlesString = $particles < 1000 ? "{$particles}" : strval((int)($particles/1000)) . " K";
        return json_encode([
            "username" => "alien-project",
            "avatar_url" => "https://alien-project.org/alien-server/logo.png",
            "content" => "",
            "embeds" => [
                [
                    "author" => [
                        "name" => $message,
                        "icon_url" => "https://alien-project.org/alien-server/galaxy.png"
                    ],
                    "title" => $simName,
                    "description" => $simDesc,
                    "fields" => [
                        [
                            "name" => "User",
                            "value" => $userName,
                            "inline" => true
                        ],
                        [
                          "name" => "Size",
                          "value" => "{$width} x {$height}",
                          "inline" => true
                        ],
                        [
                          "name" => "Objects",
                          "value" => $particlesString,
                          "inline" => true
                        ]
                    ]
                ]
            ]
        ], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    }

    function createMessageForGenome($message, $simName, $userName, $simDesc, $width, $height, $particles) {
        return json_encode([
            "username" => "alien-project",
            "avatar_url" => "https://alien-project.org/alien-server/logo.png",
            "content" => "",
            "embeds" => [
                [
                    "author" => [
                        "name" => $message,
                        "icon_url" => "https://alien-project.org/alien-server/genome.png"
                    ],
                    "title" => $simName,
                    "description" => $simDesc,
                    "fields" => [
                        [
                            "name" => "User",
                            "value" => $userName,
                            "inline" => true
                        ],
                        [
                          "name" => "Cells",
                          "value" => "{$particles}",
                          "inline" => true
                        ]
                    ]
                ]
            ]
        ], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    }
?>