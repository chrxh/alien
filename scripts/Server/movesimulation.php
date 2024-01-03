<?php
    require './helpers.php';

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
    if ($obj && strcmp($obj->userName, $userName) != 0) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    if ((int)$targetWorkspace < 0 || (int)$targetWorkspace > 2) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }
    
    if (!$db->query("UPDATE simulation SET TIMESTAMP= TIMESTAMP, FROM_RELEASE=" . addslashes($targetWorkspace) . " where ID=" . addslashes($simId))) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }
    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>