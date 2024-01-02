<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $simId = $_POST["simId"];

    if (!checkPw($db, $userName, $pw)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $obj = $db->query("SELECT u.NAME as userName FROM simulation sim, user u WHERE sim.USER_ID = u.ID and sim.ID='".addslashes($simId)."'")->fetch_object();
    if ($obj && strcmp($obj->userName, $userName) != 0) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $db->query("DELETE FROM userlike WHERE SIMULATION_ID=$simId");
    $db->query("DELETE FROM simulation WHERE ID=$simId");
    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>