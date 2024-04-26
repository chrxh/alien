<?php
    require './helpers.php';
    require './hooks.php';

    function closeAndExit($db) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }


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
    $particles = (int)$_POST['particles'];
    $version = $_POST['version'];
    $content = $_POST['content'];
    $settings = $_POST['settings'];
    $simId = $_POST['simId'];
    $size = strlen($content);
    $type = array_key_exists('type', $_POST) ? $_POST['type'] : 0;
    $workspace = array_key_exists('workspace', $_POST) ? $_POST['workspace'] : 0;
    $statistics = array_key_exists('statistics', $_POST) ? $_POST['statistics'] : "";

    if ($userName != 'alien-project' && $workspace == 1) {
        closeAndExit($db);
    }

    $stmt = $db->prepare("UPDATE simulation SET PARTICLES=?, VERSION=?, CONTENT=?, WIDTH=?, HEIGHT=?, SETTINGS=?, SIZE=?, STATISTICS=?, CONTENT2=?, CONTENT3=?, CONTENT4=?, CONTENT5=?, CONTENT6=? WHERE ID=?");
    if (!$stmt) {
        closeAndExit($db);
    }

    $emptyString = '';
    $stmt->bind_param("issiisissssssi", $particles, $version, $content, $width, $height, $settings, $size, $statistics, $emptyString, $emptyString, $emptyString, $emptyString, $emptyString, $simId);

    if (!$stmt->execute()) {
        closeAndExit($db);
    }

    // create Discord message
    //if ($workspace != PRIVATE_WORKSPACE_TYPE) {
    //    $discordPayload = createAddResourceMessage($type, $simName, $userName, $simDesc, $width, $height, $particles);
    //    sendDiscordMessage($discordPayload);
    //}

    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>