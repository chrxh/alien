<?php
    require './helpers.php';
    require './hooks.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $simId = $_POST["simId"];
    $targetWorkspace = $_POST["targetWorkspace"];

    if (!checkPw($db, $userName, $pw)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $obj = $db->query("SELECT u.NAME as userName FROM simulation sim, user u WHERE sim.USER_ID = u.ID and sim.ID=" . addslashes($simId))->fetch_object();
    if (!$obj || strcmp($obj->userName, $userName) != 0) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    if (!((int)$targetWorkspace == 0 || (int)$targetWorkspace == 2)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }
    
    if (!$db->query("UPDATE simulation SET TIMESTAMP= TIMESTAMP, FROM_RELEASE=" . addslashes($targetWorkspace) . " where ID=" . addslashes($simId))) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $obj = $db->query("SELECT sim.TYPE as type, sim.NAME as simName, u.NAME as userName, sim.DESCRIPTION as simDesc, sim.WIDTH as width, sim.HEIGHT as height, sim.particles as particles FROM simulation sim, user u WHERE sim.USER_ID=u.ID and sim.ID=" . addslashes($simId))->fetch_object();

    // create Discord message
    if ($targetWorkspace != PRIVATE_WORKSPACE_TYPE) {
        $discordPayload = createAddResourceMessage($obj->type, $obj->simName, $obj->userName, $obj->simDesc, $obj->width, $obj->height, $obj->particles);
        sendDiscordMessage($discordPayload);
    }

    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>