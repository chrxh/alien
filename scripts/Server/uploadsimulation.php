<?php
    require './helpers.php';
    require './hooks.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];

    if (!checkPw($db, $userName, $pw)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $obj = $db->query("SELECT u.ID as id FROM user u WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    if (!$obj) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $success = false;
    $simName = $_POST['simName'];
    $simDesc = $_POST['simDesc'];
    $width = (int)$_POST['width'];
    $height = (int)$_POST['height'];
    $particles = (int)$_POST['particles'];
    $version = $_POST['version'];
    $content = $_POST['content'];
    $settings = $_POST['settings'];
    $symbolMap = $_POST['symbolMap'];
    $size = strlen($content);
    $type = array_key_exists('type', $_POST) ? $_POST['type'] : 0;
    $workspace = array_key_exists('workspace', $_POST) ? $_POST['workspace'] : 0;
    $statistics = array_key_exists('statistics', $_POST) ? $_POST['statistics'] : "";

    if ($userName != 'alien-project' && $workspace == 1) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    if (!$db->query("INSERT INTO simulation (ID, USER_ID, NAME, WIDTH, HEIGHT, PARTICLES, VERSION, DESCRIPTION, CONTENT, SETTINGS, SYMBOL_MAP, PICTURE, TIMESTAMP, FROM_RELEASE, SIZE, TYPE, STATISTICS)
                    VALUES (NULL, {$obj->id}, '" . addslashes($simName) . "', $width, $height, $particles, '" . addslashes($version) . "', '" . addslashes($simDesc) . "', '" . addslashes($content) . "', '" . addslashes($settings) . "', '" . addslashes($symbolMap) . "', '" . "" . "', " . "NULL, " .addslashes($workspace) . ", " . addslashes($size) . ", " . addslashes($type) . ", '" . addslashes($statistics) . "')")) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    // create Discord message
    if ($workspace != PRIVATE_WORKSPACE_TYPE) {
        $discordPayload = createAddResourceMessage($type, $simName, $userName, $simDesc, $width, $height, $particles);
        sendDiscordMessage($discordPayload);
    }

    echo json_encode(["result"=>true, "simId"=>strval($db->insert_id)]);

    $db->commit();
    $db->close();
?>